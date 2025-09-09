from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.nl2test import Pipeline as NL2TestPipeline
from nltest.nl2test.models import NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.utils.config import init_config
from nltest.utils.llm.model import Provider


@ray.remote
class LocalizationActor:
    """
    Ray actor that initializes config and CLDK analysis once per project.
    """

    def __init__(
        self,
        *,
        project_name: str,
        base_project_dir: str,
        output_dir: str,
        llm_model: str,
        emb_model: str | None,
        llm_provider: Provider,
        llm_api_url: str | None,
        emb_provider: Provider,
        emb_api_url: str | None,
        decomposition_mode: str | DecompositionMode,
        localization_max_iters: int,
    ) -> None:
        self.project_name = project_name
        self.project_root = Path(base_project_dir) / project_name
        self.project_output_dir = Path(output_dir) / project_name
        self.llm_model = llm_model
        self.emb_model = emb_model
        self.decomposition_mode = (
            decomposition_mode
            if isinstance(decomposition_mode, DecompositionMode)
            else DecompositionMode(decomposition_mode)
        )
        self.localization_max_iters = localization_max_iters

        self.project_output_dir.mkdir(parents=True, exist_ok=True)

        init_config(
            project_name=self.project_name,
            base_project_dir=str(self.project_root),
            output_dir=str(self.project_output_dir),
            llm_provider=llm_provider,
            llm_model=self.llm_model,
            emb_provider=emb_provider,
            emb_model=self.emb_model,
            llm_api_url=llm_api_url,
            emb_api_url=emb_api_url,
            llm_api_key=os.getenv("LLM_API_KEY"),
            emb_api_key=os.getenv("EMB_API_KEY"),
            localization_max_iters=(self.localization_max_iters or 20),
        )
        self.analysis = CLDK(language="java").analysis(
            project_path=self.project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=self.project_output_dir,
            eager=False,
        )

        # Pipeline should be immutable on calls to localize
        self.pipeline = NL2TestPipeline(
            self.analysis,
            self.project_root,
            decomposition_mode=self.decomposition_mode,
            analysis_dir=self.project_output_dir,
        )
        self.pipeline.run_preprocessing()

    def localize_one(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        """Run localization evaluation for a single NL2Test input payload."""
        try:
            nl2_input = NL2TestInput(**input_payload)
            output = self.pipeline.run_localization_evaluation_pipeline(nl2_input)
            return {"success": True, "output": output.model_dump(mode="json")}
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "input": input_payload,
            }

    def localize_batch(
        self, input_payloads: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run localization evaluation for a batch of payloads sequentially in this actor."""
        results: list[dict[str, Any]] = []
        for payload in input_payloads:
            results.append(self.localize_one(payload))
        return results
