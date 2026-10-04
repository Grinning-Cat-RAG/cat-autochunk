from typing import Any, Dict

from cat import endpoint
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.exceptions import CustomValidationException

from . import state
from .optimizer import available_algorithms
from .runner import is_plugin_active
from .scheduler import get_white_rabbit, job_id_for, manual_job_id_for, schedule_now


def _job_info(job) -> Dict[str, Any] | None:
    if job is None:
        return None
    return {"id": job.id, "next_run": job.next_run, "status": str(job.status) if job.status else None}


@endpoint.get("/status", prefix="/autochunk", tags=["AutoChunk"])
async def get_status(
    info: AuthorizedInfo = check_permissions(AuthResource.CHUNKER, AuthPermission.READ),
) -> Dict[str, Any]:
    """Status of the AutoChunk plugin on the Cheshire Cat: scheduled job, running run, last report."""
    agent_id = info.cheshire_cat.agent_key
    white_rabbit = get_white_rabbit()

    return {
        "active": is_plugin_active(info.cheshire_cat),
        "scheduled_job": _job_info(white_rabbit.get_job(job_id_for(agent_id))) if white_rabbit else None,
        "manual_job": _job_info(white_rabbit.get_job(manual_job_id_for(agent_id))) if white_rabbit else None,
        "running": await state.RunLock.is_locked(agent_id),
        "last_report": await state.load_report(agent_id),
        "pending_files": (await state.load_pending(agent_id) or {}).get("files", []),
    }


@endpoint.post("/run", prefix="/autochunk", tags=["AutoChunk"])
async def run_now(
    info: AuthorizedInfo = check_permissions(AuthResource.CHUNKER, AuthPermission.WRITE),
) -> Dict[str, Any]:
    """Run the optimization of the chunker and the re-ingestion of the datalake now, in background (White Rabbit)."""
    ccat = info.cheshire_cat
    if not is_plugin_active(ccat):
        raise CustomValidationException("The AutoChunk plugin is not active on this Cheshire Cat")
    if await state.RunLock.is_locked(ccat.agent_key):
        raise CustomValidationException("A run is already in progress on this Cheshire Cat")

    job_id = schedule_now(ccat.agent_key)
    if job_id is None:
        raise CustomValidationException("White Rabbit is not available: the run cannot be scheduled")
    return {"scheduled": True, "job_id": job_id}


@endpoint.get("/algorithms", prefix="/autochunk", tags=["AutoChunk"])
async def get_algorithms(
    info: AuthorizedInfo = check_permissions(AuthResource.CHUNKER, AuthPermission.READ),
) -> Dict[str, Any]:
    """The pyVolutionary algorithms available for the setting `algorithm`."""
    return {"algorithms": available_algorithms()}
