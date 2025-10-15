from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.test2nl.pipeline import Pipeline as Test2NLPipeline
from nltest.test2nl.model.models import AbstractionLevel
from nltest.utils.config import init_config
from nltest.utils.llm.model import Provider
from nltest.utils.models import Method


@ray.remote
class Test2NLActor:
    """
    Ray actor that initializes project-scoped analysis and Test2NL pipeline once.
    """

    def __init__(
        self,
        *,
        project_name: str,
        analysis_root_dir: str,
        output_dir: str,
        llm_model: str,
        llm_provider: Provider,
        llm_api_url: str | None = None,
        base_project_dir: str | None = None,
    ) -> None:
        self.project_name = project_name
        self.analysis_dir = Path(analysis_root_dir) / project_name
        self.output_dir = Path(output_dir)
        self.llm_model = llm_model
        self.base_project_dir = (
            Path(base_project_dir) if base_project_dir is not None else Path("")
        )

        # Ensure output dir exists (even though saves happen in CLI)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize config so downstream generators pick up model/provider/temps
        init_config(
            project_name=self.project_name,
            base_project_dir=str(self.base_project_dir),
            output_dir=str(self.output_dir),
            llm_provider=llm_provider,
            llm_model=self.llm_model,
            emb_provider=None,
            emb_model=None,
            llm_api_url=llm_api_url,
            llm_api_key=os.getenv("LLM_API_KEY"),
            emb_api_key=None,
            localization_max_iters=20,
        )

        # Load analysis from precomputed JSON for this project
        self.analysis = CLDK(language="java").analysis(
            project_path="",
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=self.analysis_dir,
            eager=False,
        )

        # Instantiate the Test2NL pipeline (no side-effectful writes here)
        self.pipeline = Test2NLPipeline(
            self.analysis,
            self.project_name,
            self.output_dir,
        )

    def generate_descriptions_one(
        self, input_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate a description for a single test method.
        Expects payload with: qualified_class_name, method_signature, id (optional).
        """
        try:
            method = Method(
                qualified_class_name=input_payload["qualified_class_name"],
                method_signature=input_payload["method_signature"],
            )
            # id is optional; callers may override returned IDs when saving
            call_id = int(input_payload.get("id", 0))
            raw_abs = input_payload.get("abstraction_level")
            if raw_abs is None:
                raise ValueError("Missing required 'abstraction_level' in payload")
            if isinstance(raw_abs, AbstractionLevel):
                abs_level = raw_abs
            else:
                # Accept strings like "high"/"medium"/"low" (case-insensitive)
                abs_level = AbstractionLevel(str(raw_abs).lower())

            entry, desc = self.pipeline.run_description_of_method(
                select_method=method, id=call_id, abstraction=abs_level
            )
            if entry is None or desc is None:
                return {
                    "success": False,
                    "error": "No description generated",
                    "input": input_payload,
                }
            return {
                "success": True,
                "entry": entry.model_dump(mode="json"),
                "description": desc.model_dump(mode="json"),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "input": input_payload,
            }

    def generate_descriptions_batch(
        self, input_payloads: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Sequentially process a batch of methods within this actor."""
        results: list[dict[str, Any]] = []
        for payload in input_payloads:
            results.append(self.generate_descriptions_one(payload))
        return results
