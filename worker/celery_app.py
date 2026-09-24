"""Celery worker + beat schedule (TRD 10.1, 14.2). Every job is idempotent, so restarts and double-fires are safe."""
import os

from celery import Celery
from celery.schedules import crontab

app = Celery("inventory", broker=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
             backend=os.environ.get("REDIS_URL", "redis://redis:6379/0"))
app.conf.update(task_acks_late=True, task_reject_on_worker_lost=True, worker_prefetch_multiplier=1,
                task_time_limit=1800, timezone="UTC")
app.conf.beat_schedule = {
    "forecast-nightly": {"task": "worker.tasks.run_forecast_task", "schedule": crontab(hour=1, minute=0),
                         "args": (14, "all", None, "svc:scheduler")},
    "reorder-after-forecast": {"task": "worker.tasks.evaluate_reorders_task", "schedule": crontab(minute=30, hour="*/6")},
    "po-delay-check": {"task": "worker.tasks.po_delay_check_task", "schedule": crontab(minute=15)},
}
app.autodiscover_tasks(["worker"])
