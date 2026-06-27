"""Shared pytest fixtures. Tests are CPU-only and never download models/data:
they use the built-in mini-corpus + the BM25 / TF-IDF retrievers + identity reranker.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _artifacts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("scisearch_artifacts")
    os.environ["SCISEARCH_ARTIFACTS_DIR"] = str(d)
    os.environ.setdefault("SCISEARCH_LOG_LEVEL", "WARNING")
    yield


@pytest.fixture
def cfg():
    from scisearch.config import AppConfig
    c = AppConfig()
    c.data.use_hf_corpus = False        # offline: built-in mini-corpus
    return c
