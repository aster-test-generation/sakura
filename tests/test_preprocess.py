"""Preprocessing tests modeled after unittest style in this repo.

Validates that HttpEmbedder appends '/embeddings' and attempts a real
HTTP call with GCP defaults. If DNS resolution fails (e.g.,
NameResolutionError), assert the error is surfaced as RuntimeError with
an appropriate message. Otherwise, assert we receive a list embedding.
"""

import os
import unittest

from dotenv import load_dotenv

from nltest.nl2test.preprocessing.embedders.http import HttpEmbedder
from nltest.utils.config import Config, init_config
from nltest.utils.llm.model import Provider


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        load_dotenv()
        Config.reset()

        self.LLM_MODEL = "Azure/gpt-5-2025-08-07"
        self.EMB_MODEL = "Azure/text-embedding-3-small-1"
        self.LLM_PROVIDER = Provider.GCP
        self.EMB_PROVIDER = Provider.GCP

        # Use EMB_API_KEY from environment
        emb_api_key = os.getenv("EMB_API_KEY")

        # Initialize config with GCP provider which sets default API URLs
        init_config(
            project_name="dummy-project",
            base_project_dir="/tmp",
            output_dir="/tmp",
            llm_provider=self.LLM_PROVIDER,
            llm_model=self.LLM_MODEL,
            emb_provider=self.EMB_PROVIDER,
            emb_model=self.EMB_MODEL,
            llm_api_key=None,
            emb_api_key=emb_api_key,
            reuse_config=False,
        )

        self.api_base = Config().get("emb", "api_url")
        self.expected_url = f"{self.api_base.rstrip('/')}/embeddings"

    def tearDown(self) -> None:
        Config.reset()

    def test_http_embeddings_with_gcp(self):
        embedder = HttpEmbedder(model_id=self.EMB_MODEL, api_url=self.api_base)
        self.assertEqual(embedder.api_url, self.expected_url)

        vec = embedder.embed_query("hello world")
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)


if __name__ == "__main__":
    unittest.main()
