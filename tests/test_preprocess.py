"""Preprocessing tests modeled after unittest style in this repo.

Validates that HttpEmbedder appends '/embeddings' and works with GCP
provider defaults while mocking HTTP calls.
"""

import unittest
from unittest.mock import patch

import requests

from nltest.nl2test.preprocessing.embedders.http import HttpEmbedder
from nltest.utils.config import Config, init_config
from nltest.utils.llm.model import Provider


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the config singleton so tests are isolated.
        Config.reset()

        # Copied from scripts/run_nl2test_on_test2nl.py (relevant params only).
        self.LLM_MODEL = "Azure/gpt-5-2025-08-07"
        self.EMB_MODEL = "Azure/text-embedding-3-small-1"
        self.LLM_PROVIDER = Provider.GCP
        self.EMB_PROVIDER = Provider.GCP

        # Initialize config with GCP provider which sets default API URLs.
        init_config(
            project_name="dummy-project",
            base_project_dir="/tmp",
            output_dir="/tmp",
            llm_provider=self.LLM_PROVIDER,
            llm_model=self.LLM_MODEL,
            emb_provider=self.EMB_PROVIDER,
            emb_model=self.EMB_MODEL,
            llm_api_key=None,
            emb_api_key=None,
            reuse_config=False,
        )

        self.api_base = Config().get("emb", "api_url")
        self.expected_url = f"{self.api_base.rstrip('/')}/embeddings"

    def tearDown(self) -> None:
        Config.reset()

    def test_http_embeddings_with_gcp(self):
        """Ensure HttpEmbedder uses '<api>/embeddings' and returns vector."""

        class DummyResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        def fake_post(url, _headers=None, _json=None):
            self.assertEqual(url, self.expected_url)
            return DummyResponse()

        with patch.object(requests, "post", side_effect=fake_post):
            embedder = HttpEmbedder(model_id=self.EMB_MODEL, api_url=self.api_base)
            self.assertEqual(embedder.api_url, self.expected_url)
            vec = embedder.embed_query("hello world")
            self.assertIsInstance(vec, list)
            self.assertEqual(vec, [0.1, 0.2, 0.3])
            self.assertEqual(embedder.dim, 3)


if __name__ == "__main__":
    unittest.main()
