import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Type

from cat.db.cruds import settings as crud_settings
from cat.log import log
from cat.services.factory.chunker import ChunkerSettings, RecursiveTextChunkerSettings
from cat.services.service_factory import ServiceFactory

from . import state
from .datalake import guess_content_type, list_datalake_files, read_datalake_file
from .evaluation import (
    WORST_FITNESS,
    ChunkerEvaluator,
    EmbeddingCache,
    EvaluationResult,
    SampleFile,
    generate_questions,
    llm_reply_to_text,
    truncate_docs,
)
from .optimizer import run_optimization
from .pipeline import parse_file, reinject_file, split_with_chunker
from .search_space import Candidate, SearchSpace, build_chunker_space, canonical_key
from .settings import AutoChunkSettings

# the id of the plugin is the name of its folder (it depends on how the plugin is installed)
PLUGIN_ID = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

CHUNKER_FACTORY_PARAMS = {
    "factory_allowed_handler_name": "factory_allowed_chunkers",
    "setting_category": "chunker",
    "schema_name": "chunkerName",
}


def chunker_service_factory(ccat) -> ServiceFactory:
    return ServiceFactory(agent_key=ccat.agent_key, hook_manager=ccat.plugin_manager, **CHUNKER_FACTORY_PARAMS)


async def get_allowed_chunker_classes(ccat) -> List[Type[ChunkerSettings]]:
    """All the chunkers available to the Cheshire Cat (core ones and the ones added by its plugins)."""
    classes = await ccat.plugin_manager.execute_hook(
        "factory_allowed_chunkers", [RecursiveTextChunkerSettings], caller=None,
    )
    unique: Dict[str, Type[ChunkerSettings]] = {}
    for cls in classes or []:
        unique.setdefault(cls.__name__, cls)
    return list(unique.values())


async def get_current_chunker(ccat) -> Dict[str, Any] | None:
    """The active chunker of the Cheshire Cat, as {"name": settings class name, "value": configuration}."""
    setting = await crud_settings.get_settings_by_category(ccat.agent_key, "chunker")
    if not setting:
        return None
    return {"name": setting["name"], "value": setting.get("value") or {}}


async def load_plugin_settings(ccat) -> AutoChunkSettings:
    plugin = ccat.plugin_manager.plugins[PLUGIN_ID]
    return AutoChunkSettings(**(await plugin.load_settings(ccat.agent_key) or {}))


def is_plugin_active(ccat) -> bool:
    return PLUGIN_ID in ccat.plugin_manager.active_plugins


async def _ask_llm(ccat, prompt: str) -> str:
    return llm_reply_to_text(await ccat.large_language_model.ainvoke(prompt))


async def build_samples(ccat, files: List[str], settings: AutoChunkSettings, rng: random.Random) -> List[SampleFile]:
    """Parse a random sample of the datalake files (not split yet)."""
    candidates = list(files)
    rng.shuffle(candidates)

    samples = []
    for name in candidates:
        if len(samples) >= settings.sample_max_files:
            break
        content = read_datalake_file(ccat, name)
        if not content:
            continue
        try:
            docs = await parse_file(ccat, name, content, guess_content_type(name, content))
        except Exception as e:
            log.warning(f"AutoChunk: unable to parse {name}, excluded from the sample: {e}")
            continue
        docs = truncate_docs(docs, settings.sample_max_chars_per_file)
        if docs:
            samples.append(SampleFile(name=name, docs=docs))
    return samples


class AutoChunkRun:
    """A run of the plugin on a Cheshire Cat: optimization of the chunker, then re-ingestion of the datalake."""

    def __init__(self, ccat, settings: AutoChunkSettings, trigger: str):
        self.ccat = ccat
        self.settings = settings
        self.rng = random.Random(settings.seed)
        self.report: Dict[str, Any] = {
            "agent_id": ccat.agent_key,
            "trigger": trigger,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "algorithm": settings.algorithm,
        }

    def _finish(self, status: str, message: str | None = None) -> Dict[str, Any]:
        self.report["status"] = status
        if message:
            self.report["message"] = message
        self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
        log.info(f"AutoChunk: run on agent {self.ccat.agent_key} finished: {status}" + (f" - {message}" if message else ""))
        return self.report

    async def _evaluate(self, evaluator: ChunkerEvaluator, candidate: Candidate) -> EvaluationResult:
        try:
            candidate.settings_class.model_validate(candidate.config)
            chunker = candidate.settings_class.get_from_config(candidate.config)
        except Exception as e:
            return EvaluationResult(fitness=WORST_FITNESS, error=f"invalid configuration: {e}")
        try:
            return await evaluator.evaluate(chunker)
        except Exception as e:
            return EvaluationResult(fitness=WORST_FITNESS, error=str(e))

    async def optimize(self, files: List[str], current: Dict[str, Any] | None) -> Candidate | None:
        """Find the best chunker configuration. Returns None when the current chunker has to be kept."""
        settings = self.settings

        classes = await get_allowed_chunker_classes(self.ccat)
        if settings.allowed_chunkers:
            classes = [c for c in classes if c.__name__ in settings.allowed_chunkers]
        current_name = current["name"] if current else None
        current_config = current["value"] if current else None

        spaces = [
            space for cls in classes
            if (space := build_chunker_space(cls, current_name, current_config, settings.search_space_overrides))
        ]
        if not spaces:
            self.report["message"] = "no chunker can be configured"
            return None
        space = SearchSpace(spaces)
        self.report["search_space"] = {
            "chunkers": [s.name for s in spaces],
            "variables": space.variable_names,
        }

        samples = await build_samples(self.ccat, files, settings, self.rng)
        self.report["sample_files"] = [s.name for s in samples]
        if not samples:
            self.report["message"] = "no file of the datalake can be parsed"
            return None

        questions = await generate_questions(
            samples, lambda prompt: _ask_llm(self.ccat, prompt), settings.questions_per_file, self.rng,
        )
        self.report["questions"] = len(questions)
        if len(questions) < settings.min_questions:
            self.report["message"] = (
                f"only {len(questions)} valid synthetic questions generated (minimum {settings.min_questions}): "
                f"is the LLM of the Cheshire Cat configured?"
            )
            return None

        embedder = await self.ccat.embedder()
        cache = EmbeddingCache(embed_documents=lambda texts: self.ccat._run_in_ingestion_executor(
            embedder.embed_documents, texts
        ))
        evaluator = ChunkerEvaluator(
            samples=samples,
            questions=questions,
            cache=cache,
            split=lambda chunker, docs: split_with_chunker(self.ccat, chunker, docs),
            top_k=settings.top_k,
            size_penalty=settings.size_penalty,
        )

        baseline = None
        current_class = next((s.settings_class for s in spaces if s.name == current_name), None)
        if current_class is None and current_name:
            # the active chunker is not in the search space (e.g. excluded by allowed_chunkers): still the reference
            current_class = next((c for c in await get_allowed_chunker_classes(self.ccat) if c.__name__ == current_name), None)
        if current_class is not None:
            try:
                baseline = Candidate(
                    settings_class=current_class,
                    config=current_class(**current_class.parse_config(current_config or {})).model_dump(mode="json"),
                )
            except Exception as e:
                log.warning(f"AutoChunk: the current chunker configuration is not valid: {e}")

        outcome = await run_optimization(
            space=space,
            evaluate=lambda candidate: self._evaluate(evaluator, candidate),
            algorithm=settings.algorithm,
            algorithm_parameters=settings.algorithm_parameters,
            population_size=settings.population_size,
            max_cycles=settings.max_cycles,
            max_evaluations=settings.max_evaluations,
            max_runtime_seconds=settings.max_runtime_minutes * 60,
            seed=settings.seed,
            baseline=baseline,
        )

        baseline_result = next((e.result for e in outcome.evaluations if baseline and e.candidate.key == baseline.key), None)
        best = outcome.best
        self.report["optimization"] = {
            "evaluations": len(outcome.evaluations),
            "budget_exhausted": outcome.budget_exhausted,
            "error": outcome.error,
            "baseline": {
                "chunker": baseline.name, "config": baseline.config, **baseline_result.as_dict(),
            } if baseline and baseline_result else None,
            "best": {"chunker": best.candidate.name, "config": best.candidate.config, **best.result.as_dict()} if best else None,
        }

        if best is None:
            self.report["message"] = "no valid chunker configuration found"
            return None

        if baseline is not None and best.candidate.key == baseline.key:
            self.report["message"] = "the current chunker is already the best one"
            return None

        if baseline_result is not None and baseline_result.error is None:
            gain = best.result.fitness - baseline_result.fitness
            if gain < settings.min_improvement:
                self.report["message"] = (
                    f"improvement {gain:.4f} below the minimum ({settings.min_improvement}): the current chunker is kept"
                )
                return None

        return best.candidate

    async def activate_chunker(self, candidate: Candidate) -> None:
        """Make the candidate the active chunker of the Cheshire Cat (as the chunker settings endpoint does)."""
        await chunker_service_factory(self.ccat).upsert_service(candidate.name, candidate.config)
        chunker = await self.ccat.service_provider.get_chunker(self.ccat.agent_key, self.ccat.plugin_manager)
        if type(chunker) is not candidate.settings_class.pyclass():
            raise RuntimeError(f"the chunker {candidate.name} cannot be instantiated by the factory")
        self.ccat.chunker = chunker

    async def reinject(self, files: List[str], chunker_key: str) -> List[str]:
        """Re-ingest the files with the active chunker. Returns the files whose re-ingestion failed."""
        outcomes = []
        for name in files:
            outcome = await reinject_file(self.ccat, name)
            outcomes.append(outcome)
            log.info(
                f"AutoChunk: re-ingestion of {name}: "
                + (f"{outcome.old_points} -> {outcome.new_points} points" if outcome.success else f"failed ({outcome.error})")
            )

        failed = [o.name for o in outcomes if not o.success]
        self.report["reinjection"] = {
            "files": len(outcomes),
            "succeeded": len(outcomes) - len(failed),
            "failed": [o.as_dict() for o in outcomes if not o.success],
            "old_points": sum(o.old_points for o in outcomes if o.success),
            "new_points": sum(o.new_points for o in outcomes if o.success),
        }
        await state.save_pending(self.ccat.agent_key, {"chunker": chunker_key, "files": failed})
        return failed

    async def execute(self) -> Dict[str, Any]:
        started = time.monotonic()
        files = await list_datalake_files(self.ccat)
        self.report["datalake_files"] = len(files)
        if not files:
            return self._finish("skipped", "the datalake is empty")

        current = await get_current_chunker(self.ccat)
        self.report["previous_chunker"] = current

        best = await self.optimize(files, current)
        self.report["optimization_seconds"] = round(time.monotonic() - started, 1)

        if best is None:
            # the chunker does not change: only the files whose last re-ingestion failed are retried
            pending = await state.load_pending(self.ccat.agent_key)
            current_key = canonical_key(current["name"], current["value"]) if current else None
            retry = [f for f in (pending or {}).get("files", []) if f in files]
            if retry and pending.get("chunker") == current_key:
                await self.reinject(retry, current_key)  # type: ignore[arg-type]
                return self._finish("retried", self.report.get("message"))
            return self._finish("unchanged", self.report.get("message"))

        try:
            await self.activate_chunker(best)
        except Exception as e:
            if current:
                # restore the previous chunker
                try:
                    await chunker_service_factory(self.ccat).upsert_service(current["name"], current["value"])
                except Exception as restore_error:
                    log.error(f"AutoChunk: unable to restore the previous chunker: {restore_error}")
            return self._finish("failed", f"unable to activate the chunker {best.name}: {e}")

        self.report["new_chunker"] = {"name": best.name, "value": best.config}
        failed = await self.reinject(files, best.key)
        status = "completed" if not failed else "completed_with_errors"
        return self._finish(status, f"chunker changed to {best.name}, {len(files) - len(failed)}/{len(files)} files re-ingested")


async def run_for_agent(agent_id: str, trigger: str = "schedule", force: bool = False) -> Dict[str, Any]:
    """Run the optimization and the re-ingestion for a Cheshire Cat.

    Args:
        agent_id: the id of the Cheshire Cat.
        trigger: what started the run ("schedule" or "manual"), for the report.
        force: run even if the plugin is disabled in its settings.
    """
    from cat.looking_glass.bill_the_lizard import BillTheLizard

    ccat = await BillTheLizard().get_cheshire_cat(agent_id)
    if ccat is None:
        return {"agent_id": agent_id, "status": "skipped", "message": "unknown agent"}
    if not is_plugin_active(ccat):
        return {"agent_id": agent_id, "status": "skipped", "message": "the plugin is not active on the agent"}

    settings = await load_plugin_settings(ccat)
    if not settings.enabled and not force:
        return {"agent_id": agent_id, "status": "skipped", "message": "disabled in the settings of the plugin"}

    lock = state.RunLock(agent_id, settings.lock_ttl_minutes * 60)
    if not await lock.acquire():
        return {"agent_id": agent_id, "status": "skipped", "message": "a run is already in progress"}

    run = AutoChunkRun(ccat, settings, trigger)
    try:
        await state.save_report(agent_id, run.report)
        report = await run.execute()
    except Exception as e:
        log.error(f"AutoChunk: run on agent {agent_id} failed: {e}")
        report = run._finish("failed", str(e))
    finally:
        await lock.release()

    await state.save_report(agent_id, report)
    return report
