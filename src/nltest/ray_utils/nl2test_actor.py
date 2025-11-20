from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import logging
import traceback

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.nl2test import Pipeline as NL2TestPipeline
from nltest.nl2test.models import NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.utils.config import init_config
from nltest.utils.llm.model import Provider
from nltest.utils.pretty.color_logger import RichLog


@ray.remote
class NL2TestActor:
    """
    Ray actor that initializes config, CLDK analysis, and pipeline once per project.
    """

    def __init__(
        self,
        *,
        project_name: str,
        base_project_dir: str,
        base_analysis_dir: str,
        output_dir: str,
        llm_model: str,
        emb_model: str | None,
        llm_provider: Provider | None,
        llm_api_url: str | None,
        emb_provider: Provider | None,
        emb_api_url: str | None,
        decomposition_mode: str | DecompositionMode,
        supervisor_max_iters: int,
        localization_max_iters: int,
        composition_max_iters: int,
        can_parallel_tool: bool = True,
        use_stored_index: bool = True,
        debug: bool = False,
        log_file_name: str | None = None,
        exclude_test_dirs: bool = False,
    ) -> None:
        self.project_name = project_name
        self.project_root = Path(base_project_dir) / project_name
        self.analysis_dir = Path(base_analysis_dir) / project_name
        self.project_output_dir = Path(output_dir) / project_name

        self.llm_model = llm_model
        self.emb_model = emb_model
        self.decomposition_mode = (
            decomposition_mode
            if isinstance(decomposition_mode, DecompositionMode)
            else DecompositionMode(decomposition_mode)
        )

        # Ensure output dir exists (per-project)
        self.project_output_dir.mkdir(parents=True, exist_ok=True)

        if debug:
            RichLog.set_level(logging.DEBUG)
        if log_file_name:
            actor_log_path = self.project_output_dir / log_file_name
            try:
                RichLog.add_file_handler(str(actor_log_path), overwrite=True)
                RichLog.info(
                    f"[NL2TestActor:{self.project_name}] Writing logs to file: {actor_log_path}"
                )
            except Exception as exc:
                RichLog.warn(
                    f"[NL2TestActor:{self.project_name}] Failed to add log file handler at {actor_log_path}: {exc}"
                )

        # Initialize shared config for this project
        init_config(
            project_name=self.project_name,
            base_project_dir=str(self.project_root),
            project_output_dir=str(self.project_output_dir),
            llm_provider=llm_provider,
            llm_model=self.llm_model,
            emb_provider=emb_provider,
            emb_model=self.emb_model,
            llm_api_url=llm_api_url,
            emb_api_url=emb_api_url,
            llm_api_key=os.getenv("LLM_API_KEY"),
            emb_api_key=os.getenv("EMB_API_KEY"),
            localization_max_iters=(localization_max_iters or 40),
            composition_max_iters=(composition_max_iters or 30),
            supervisor_max_iters=(supervisor_max_iters or 10),
            can_parallel_tool=can_parallel_tool,
            use_stored_index=use_stored_index,
        )

        # Load analysis from precomputed JSON for this project
        self.analysis = CLDK(language="java").analysis(
            project_path=str(self.project_root),
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=self.analysis_dir,
            eager=False,
        )

        # Instantiate the NL2Test pipeline
        self.pipeline = NL2TestPipeline(
            self.analysis,
            project_root=self.project_root,
            analysis_dir=self.analysis_dir,
            decomposition_mode=self.decomposition_mode,
        )
        self.pipeline.run_preprocessing(exclude_test_dirs=exclude_test_dirs)

    def run_nl2test_one(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        """Run NL2Test for a single input payload.

        Returns a dict with either {success: True, result: NL2TestEval}
        or {success: False, error: str, error_type: str, traceback: str, input: dict}.
        """
        try:
            nl2_input = NL2TestInput(**input_payload)
            result = self.pipeline.run_nl2test(nl2_input)
            return {"success": True, "result": result}
        except Exception as e:
            # Capture full traceback for easier debugging back on the driver.
            tb = traceback.format_exc()
            # Also log in the worker process so Ray log files contain details.
            RichLog.error(
                f"[NL2TestActor:{self.project_name}] Failed for input id={input_payload.get('id')} "
                f"({input_payload.get('qualified_class_name')}::{input_payload.get('method_signature')}): {e}\n{tb}"
            )
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": tb,
                "input": input_payload,
            }

    def run_nl2test_batch(
        self, input_payloads: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process a list of inputs sequentially within this actor.

        Ray handles parallelism across actors; this method keeps per-actor
        behavior simple and logs basic progress.
        """
        results: list[dict[str, Any]] = []
        for idx, payload in enumerate(input_payloads, start=1):
            RichLog.debug(
                f"[NL2TestActor:{self.project_name}] Running {idx}/{len(input_payloads)}: "
                f"{payload.get('qualified_class_name')}::{payload.get('method_signature')} (id={payload.get('id')})"
            )
            results.append(self.run_nl2test_one(payload))
        return results
