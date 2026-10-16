"""Real background scheduler (APScheduler) — the digest fires on a cron schedule, no button needed."""
import logging
import os

log = logging.getLogger("finpilot.scheduler")


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from app import digest
    sched = BackgroundScheduler(timezone=os.getenv("TZ", "Asia/Kolkata"))
    trigger = CronTrigger(day_of_week=os.getenv("DIGEST_CRON_DAY_OF_WEEK", "mon"),
                          hour=int(os.getenv("DIGEST_CRON_HOUR", "9")), minute=0)
    sched.add_job(lambda: log.info("digest: %s", digest.build_and_send().get("channel")),
                  trigger, id="weekly_digest", replace_existing=True)
    sched.start()
    log.info("scheduler started: %s", trigger)
    return sched
