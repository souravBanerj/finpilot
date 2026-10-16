"""Simple UI.  Run API first (uvicorn app.main:app) then:  streamlit run ui/streamlit_app.py"""
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("FINPILOT_API", "http://localhost:8000")
st.set_page_config(page_title="FinPilot", layout="wide")
st.title("FinPilot — forward-looking money copilot")


def call(method, path, **kw):
    r = requests.request(method, API + path, timeout=120, **kw)
    if not r.ok:
        st.error(r.text); return None
    return r.json()


with st.sidebar:
    st.header("Data")
    f = st.file_uploader("Upload transactions CSV", type="csv")
    if f and st.button("Ingest & analyze"):
        st.json(call("POST", "/upload", files={"file": f.getvalue()}))
    if st.button("Load demo persona"):
        st.json(call("POST", "/demo/load", params={"seed": 1, "opening_balance": 0}))
    bal = st.number_input("Opening balance", value=0.0)
    if st.button("Save balance"):
        call("POST", "/settings/balance", params={"opening_balance": bal})

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Insights", "Forecast & what-if", "Goals", "Ask FinPilot", "Why? (audit trail)"])

with tab1:
    data = call("GET", "/insights")
    if data and data.get("findings") is not None:
        for fnd in data["findings"]:
            st.markdown(f"**[{fnd['severity'].upper()}]** {fnd['title']}")
            st.caption(str(fnd["evidence"]))
        st.subheader("Profile"); st.json(data["profile"])

with tab2:
    fc = call("GET", "/forecast")
    if fc and "trajectory" in fc:
        t = pd.DataFrame(fc["trajectory"]).set_index("month")
        st.line_chart(t[["balance_low", "balance_base", "balance_high"]])
        st.caption(fc["assumptions"])
        st.subheader("Can I afford it?")
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Amount", value=30000.0)
        off = c2.number_input("Months from now (1 = first forecast month)", 1, 6, 2)
        nm = c3.text_input("What", "trip")
        if st.button("Simulate"):
            res = call("POST", "/simulate/purchase", json={"amount": amt, "name": nm, "month_offset": int(off)})
            if res:
                {"yes": st.success, "tight": st.warning, "no": st.error}[res["verdict"]](res["explanation"])
                st.line_chart(pd.DataFrame({"balance_after_purchase": res["balance_after"]}))
                st.json(res)

with tab3:
    with st.form("goal"):
        n = st.text_input("Goal name", "Goa trip"); tg = st.number_input("Target", value=100000.0)
        dl = st.text_input("Deadline (YYYY-MM-DD)", "2027-06-30"); sv = st.number_input("Saved so far", value=0.0)
        if st.form_submit_button("Add goal"):
            call("POST", "/goals", json={"name": n, "target_amount": tg, "deadline": dl, "saved_so_far": sv})
    for g in call("GET", "/goals") or []:
        st.write(g)

with tab4:
    q = st.text_input("Ask anything (e.g. 'Can I afford a ₹50,000 trip next month?')")
    if q:
        r = call("POST", "/chat", json={"question": q})
        if r:
            st.write(r["answer"]); st.caption(f"route: {r['route']}")

with tab5:
    if st.button("Run advisor now"):
        call("POST", "/recommendations/run")
    for d in call("GET", "/recommendations") or []:
        with st.expander(f"{d['title']}  —  {d['rec_type']}"):
            st.write(d["action"]); st.markdown(f"**Why:** {d['reason']}"); st.caption(f"trigger: `{d['trigger_ref']}`  ·  agent: {d['agent']}")
            st.json(d["evidence"])
