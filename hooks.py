from typing import Any, Dict

from cat import hook, log
from cat.db.cruds import plugins as crud_plugins, settings as crud_settings

from . import state
from .runner import PLUGIN_ID
from .scheduler import schedule, unschedule
from .settings import AutoChunkSettings


def _schedule_from_settings(agent_id: str, settings: Dict[str, Any] | None) -> None:
    try:
        schedule(agent_id, AutoChunkSettings(**(settings or {})))
    except Exception as e:
        log.error(f"AutoChunk: unable to schedule the job of agent {agent_id}: {e}")


@hook(priority=0)
async def after_lizard_bootstrap(lizard) -> None:
    """At startup, align the scheduled jobs with the agents having the plugin active (White Rabbit is created by its
    own after_lizard_bootstrap hook, with a higher priority)."""
    if getattr(lizard, "white_rabbit", None) is None:
        return

    for agent_id in await crud_settings.get_agents_main_keys():
        try:
            if PLUGIN_ID in (await crud_plugins.get_active_plugins_from_db(agent_id) or []):
                _schedule_from_settings(agent_id, await crud_plugins.get_setting(agent_id, PLUGIN_ID))
            else:
                unschedule(agent_id)
        except Exception as e:
            log.error(f"AutoChunk: unable to align the job of agent {agent_id}: {e}")


@hook(priority=0)
async def after_plugin_toggling_on_agent(plugin_id: str, cat) -> None:
    if plugin_id != PLUGIN_ID:
        return

    if plugin_id in cat.plugin_manager.active_plugins:
        plugin = cat.plugin_manager.plugins[PLUGIN_ID]
        _schedule_from_settings(cat.agent_key, await plugin.load_settings(cat.agent_key))
    else:
        unschedule(cat.agent_key)


@hook(priority=0)
def after_plugin_settings_update(plugin_id: str, settings: Dict[str, Any], cat) -> None:
    if plugin_id != PLUGIN_ID or plugin_id not in cat.plugin_manager.active_plugins:
        return
    _schedule_from_settings(cat.agent_key, settings)


@hook(priority=0)
async def after_cheshire_cat_destroy(agent_id: str, cat) -> None:
    unschedule(agent_id)
    await state.clear_agent_state(agent_id)
