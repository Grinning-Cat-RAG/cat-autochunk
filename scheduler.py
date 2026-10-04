from cat.log import log

from .cron import parse_cron_expression
from .runner import run_for_agent
from .settings import AutoChunkSettings


def job_id_for(agent_id: str) -> str:
    return f"autochunk:{agent_id}"


def manual_job_id_for(agent_id: str) -> str:
    return f"autochunk:{agent_id}:manual"


# IMPORTANT: the jobs MUST live at module level, so that APScheduler (White Rabbit) can serialize them in its Redis job
# store by their import path. All the runtime context is passed through the kwargs.
async def autochunk_job(agent_id: str, force: bool = False):
    """Scheduled job: optimize the chunker of the Cheshire Cat and re-ingest its datalake."""
    report = await run_for_agent(agent_id, trigger="manual" if force else "schedule", force=force)
    log.info(f"AutoChunk: job on agent {agent_id} ended with status {report.get('status')}")


def get_white_rabbit():
    from cat.looking_glass.bill_the_lizard import BillTheLizard

    return getattr(BillTheLizard(), "white_rabbit", None)


def unschedule(agent_id: str) -> None:
    white_rabbit = get_white_rabbit()
    if white_rabbit is None:
        return
    if white_rabbit.get_job(job_id_for(agent_id)):
        white_rabbit.remove_job(job_id_for(agent_id))


def schedule(agent_id: str, settings: AutoChunkSettings) -> str | None:
    """(Re)schedule the cron job of the Cheshire Cat according to the settings of the plugin.

    Returns:
        The id of the job, None if no job is scheduled (plugin disabled or White Rabbit not available).
    """
    white_rabbit = get_white_rabbit()
    if white_rabbit is None:
        log.warning("AutoChunk: White Rabbit is not available, the job cannot be scheduled")
        return None

    unschedule(agent_id)
    if not settings.enabled:
        return None

    cron = parse_cron_expression(settings.cron_expression)
    job_id = job_id_for(agent_id)
    try:
        return white_rabbit.schedule_cron_job(job=autochunk_job, job_id=job_id, agent_id=agent_id, **cron)
    except Exception as e:
        # another worker sharing the job store scheduled it meanwhile: replace it
        log.debug(f"AutoChunk: rescheduling {job_id} after a conflict: {e}")
        unschedule(agent_id)
        return white_rabbit.schedule_cron_job(job=autochunk_job, job_id=job_id, agent_id=agent_id, **cron)


def schedule_now(agent_id: str) -> str | None:
    """Schedule a one-shot run, as soon as possible (manual trigger), even if the plugin is disabled in its settings."""
    white_rabbit = get_white_rabbit()
    if white_rabbit is None:
        return None

    job_id = manual_job_id_for(agent_id)
    if white_rabbit.get_job(job_id):
        white_rabbit.remove_job(job_id)
    return white_rabbit.schedule_job(job=autochunk_job, job_id=job_id, seconds=1, agent_id=agent_id, force=True)
