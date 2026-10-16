import os
import tempfile
import unittest

import pandas as pd

from app import config, db
from app.ingest import parse_transactions_csv
from agents import advisor, chat, forecaster
from agents.analyzer import analyze
from agents.categorizer import categorize, normalize_merchant
from eval.generate_data import generate_persona


def cat_persona(seed=3):
    df, truth = generate_persona(seed)
    out = categorize(df[["id", "date", "description", "amount"]], use_llm=False)
    return df, out, truth


class TempDBCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = config.DB_PATH
        config.DB_PATH = os.path.join(self.tmp.name, "t.db")
        db.init_db()

    def tearDown(self):
        config.DB_PATH = self._old
        self.tmp.cleanup()


class TestCategorizer(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_merchant("UPI-SWIGGY ORDER 88231@okaxis"), "SWIGGY")
        self.assertEqual(normalize_merchant("POS AMAZON PRIME VIDEO 123"), "AMAZON PRIME")

    def test_rules_are_ordered_specific_first(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"] * 3), "amount": [-199, -500, -800],
                           "description": ["AMAZON PRIME", "AMAZON", "SWIGGY INSTAMART"]})
        self.assertEqual(categorize(df, use_llm=False)["category"].tolist(), ["Subscriptions", "Shopping", "Groceries"])

    def test_unknown_recurring_merchant_becomes_subscription(self):
        dates = pd.to_datetime(["2026-01-05", "2026-02-05", "2026-03-06", "2026-04-05"])
        df = pd.DataFrame({"date": dates, "description": ["ZENITH LEARN 123"] * 4, "amount": [-499.0] * 4})
        out = categorize(df, use_llm=False)
        self.assertTrue((out["category"] == "Subscriptions").all())
        self.assertTrue((out["method"] == "recurrence").all())

    def test_memory_overrides_rules(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "description": ["SWIGGY"], "amount": [-300.0]})
        out = categorize(df, memory={"SWIGGY": "Groceries"}, use_llm=False)
        self.assertEqual(out["category"].iloc[0], "Groceries")
        self.assertEqual(out["method"].iloc[0], "memory")

    def test_positive_amount_is_income(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "description": ["ANYTHING"], "amount": [5000.0]})
        self.assertEqual(categorize(df, use_llm=False)["category"].iloc[0], "Income")


class TestAnalyzer(unittest.TestCase):
    def test_finds_injected_patterns(self):
        df, out, truth = cat_persona(3)
        out["id"] = df["id"].values
        types = {f["type"] for f in analyze(out)["findings"]}
        self.assertIn("subscription_overlap", types)
        self.assertIn("duplicate_charge", types)
        self.assertIn("txn_outlier", types)
        self.assertIn("subscription_price_hike", types)

    def test_findings_have_stable_ids_and_evidence(self):
        df, out, _ = cat_persona(3)
        out["id"] = df["id"].values
        for f in analyze(out)["findings"]:
            self.assertTrue(f["id"]); self.assertTrue(f["evidence"]); self.assertIn(f["severity"], ("high", "medium", "low"))


class TestForecastAndSimulation(unittest.TestCase):
    def setUp(self):
        _, self.out, _ = cat_persona(3)
        raw = forecaster.forecast(self.out, 6, 0.0)
        # start the user with exactly 100,000 in the bank
        self.fc = forecaster.forecast(self.out, 6, 100_000 - raw["start_balance"])

    def test_trajectory_shape(self):
        t = self.fc["trajectory"]
        self.assertEqual(len(t), 6)
        for r in t:
            self.assertLessEqual(r["balance_low"], r["balance_base"]); self.assertLessEqual(r["balance_base"], r["balance_high"])

    def test_small_purchase_yes_huge_purchase_no(self):
        self.assertEqual(forecaster.simulate_purchase(self.fc, 1_000, 1)["verdict"], "yes")
        big = forecaster.simulate_purchase(self.fc, 400_000, 1)
        self.assertEqual(big["verdict"], "no")
        self.assertGreater(big["monthly_cut_to_make_it_work"], 0)

    def test_purchase_only_lowers_balance_from_purchase_month(self):
        r = forecaster.simulate_purchase(self.fc, 20_000, 2)
        before = [t["balance_base"] for t in self.fc["trajectory"]]
        after = r["balance_after"]
        self.assertEqual(after[0], before[0])                       # month 1 untouched
        for b, a in zip(before[1:], after[1:]):
            self.assertEqual(b - a, 20_000)                         # every later month is exactly 20k lower

    def test_goal_math(self):
        g = forecaster.simulate_goal(self.fc, target=10_000_000, deadline="2027-03-31")
        self.assertFalse(g["on_track"]); self.assertGreater(g["extra_to_save_per_month"], 0)
        easy = forecaster.simulate_goal(self.fc, target=1000, deadline="2027-03-31")
        self.assertTrue(easy["on_track"])

    def test_needs_three_months(self):
        short = self.out[self.out["date"] < "2025-12-01"]
        self.assertIn("error", forecaster.forecast(short))

    def test_model_beats_naive_in_backtest(self):
        res = [forecaster.backtest(cat_persona(s)[1]) for s in range(5)]
        model = sum(r["total_expense_mape_model"] for r in res)
        naive = sum(r["total_expense_mape_naive_last"] for r in res)
        self.assertLess(model, naive)


class TestAdvisor(unittest.TestCase):
    def test_grounding_guard(self):
        facts = "Cancel NETFLIX saves 799 per month 9588 per year"
        self.assertTrue(advisor.numbers_grounded("Drop Netflix and save 799 a month!", facts))
        self.assertFalse(advisor.numbers_grounded("Drop Netflix and save 1,500 a month!", facts))

    def test_every_rec_has_reason_and_trigger(self):
        df, out, _ = cat_persona(3)
        out["id"] = df["id"].values
        a = analyze(out)
        fc = forecaster.forecast(out)
        recs = advisor.advise(a, fc, [])
        self.assertTrue(recs)
        for r in recs:
            self.assertTrue(r["reason"]); self.assertTrue(r["trigger_ref"]); self.assertIn("evidence", r)

    def test_goal_advice_adapts(self):
        df, out, _ = cat_persona(3)
        out["id"] = df["id"].values
        a, fc = analyze(out), forecaster.forecast(out)
        today = pd.Timestamp("2026-10-14")
        ok = forecaster.goal_status({"name": "ok", "target_amount": 1000, "saved_so_far": 1000, "deadline": "2027-03-31",
                                     "start_date": "2026-07-01"}, fc, today)
        hard = forecaster.goal_status({"name": "hard", "target_amount": 50_000_000, "saved_so_far": 0, "deadline": "2027-03-31",
                                       "start_date": "2026-07-01"}, fc, today)
        refs = [r["trigger_ref"] for r in advisor.advise(a, fc, [ok, hard])]
        self.assertIn("goal:hard", refs); self.assertNotIn("goal:ok", refs)


class TestIngest(unittest.TestCase):
    def test_debit_credit_columns_and_bad_rows(self):
        csv = "Txn Date,Narration,Debit,Credit\n15/01/2026,UPI-SWIGGY 1,450,\n16/01/2026,SALARY,,50000\nbad,xx,1,\n"
        rows, rep = parse_transactions_csv(csv)
        self.assertEqual(rep["rows_ok"], 2); self.assertEqual(rep["rows_skipped"], 1)
        self.assertEqual(rows[0]["amount"], -450.0); self.assertEqual(rows[0]["date"], "2026-01-15")
        self.assertEqual(rows[1]["amount"], 50000.0)

    def test_signed_amount_csv(self):
        rows, _ = parse_transactions_csv("date,description,amount\n2026-01-02,NETFLIX,-649\n")
        self.assertEqual(rows[0]["amount"], -649.0)


class TestChat(unittest.TestCase):
    def test_parse_amount(self):
        self.assertEqual(chat.parse_amount("Can I afford a ₹50,000 trip next month?"), 50000)
        self.assertEqual(chat.parse_amount("can i buy a 1.5 lakh bike"), 150000)
        self.assertEqual(chat.parse_amount("laptop for 80k"), 80000)
        self.assertIsNone(chat.parse_amount("can I afford a trip in 3 months"))


class TestEndToEnd(TempDBCase):
    def _load(self):
        from app import coordinator
        df, _ = generate_persona(4)
        db.insert_transactions([{"date": r.date.strftime("%Y-%m-%d"), "description": r.description, "amount": r.amount}
                                for r in df.itertuples()])
        return coordinator

    def test_pipeline_runs_all_agents_and_persists_audit_trail(self):
        coordinator = self._load()
        st = coordinator.run_pipeline("demo", use_llm=False)
        self.assertEqual([t["agent"] for t in st["trace"]], ["categorizer", "analyzer", "forecaster", "goal_tracker", "advisor"])
        stored = db.load_transactions()
        self.assertFalse(stored["category"].isna().any())
        decisions = db.list_decisions()
        self.assertTrue(decisions)
        for d in decisions:
            self.assertTrue(d["reason"]); self.assertTrue(d["trigger_ref"])

    def test_rerun_does_not_duplicate_decisions(self):
        coordinator = self._load()
        coordinator.run_pipeline("demo", use_llm=False)
        n1 = len(db.list_decisions(limit=1000))
        coordinator.run_pipeline("demo", use_llm=False)
        self.assertEqual(n1, len(db.list_decisions(limit=1000)))

    def test_simulation_is_logged_and_chat_routes_to_simulator(self):
        coordinator = self._load()
        res = coordinator.simulate("demo", "purchase", amount=5000, name="headphones", month_offset=2)
        self.assertIn(res["verdict"], ("yes", "tight", "no"))
        self.assertTrue(any(d["rec_type"] == "simulation" for d in db.list_decisions()))
        ans = coordinator.handle("chat", "demo", question="Can I afford a ₹20,000 trip next month?")
        self.assertEqual(ans["route"], "simulation")

    def test_user_correction_teaches_categorizer(self):
        coordinator = self._load()
        db.remember("SWIGGY", "Groceries")
        coordinator.run_pipeline("demo", use_llm=False)
        d = db.load_transactions()
        self.assertTrue((d[d["merchant"] == "SWIGGY"]["category"] == "Groceries").all())


class TestAccuracyRegression(unittest.TestCase):
    """Keeps the README numbers honest: fails if a change degrades categorization badly."""
    def test_categorization_accuracy_floor(self):
        ok = tot = 0
        for s in range(5):
            df, out, _ = cat_persona(s)
            ok += (df["true_category"].values == out["category"].values).sum(); tot += len(df)
        self.assertGreater(ok / tot, 0.88)


if __name__ == "__main__":
    unittest.main()
