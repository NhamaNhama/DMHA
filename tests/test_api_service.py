from fastapi.testclient import TestClient
import os
import torch
from app.api_service import APIService
from app.context_system import ContextAwareSystem
from app.config import Config
from unittest.mock import patch, MagicMock
from app.utils.torchscript_utils import load_and_torchscript_model
from transformers import AutoModel, PreTrainedModel
from prometheus_client import REGISTRY, CollectorRegistry
import pytest


@pytest.fixture
def test_config():
    """テスト用の設定を提供"""
    config = Config()
    config.redis_host = "localhost"
    config.milvus_host = "localhost"
    config.use_gpu = False
    config.model_name = "distilbert-base-uncased"
    config.jwt_secret = "test_secret_key"
    config.test_mode = True
    return config


@pytest.fixture
def mock_environment(monkeypatch):
    """テスト環境用の設定を一時的に適用するfixture"""
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("SKIP_HF", "true")
    monkeypatch.setenv("SKIP_MILVUS", "true")
    yield


@pytest.fixture
def mock_model():
    """モデルのモックを提供"""
    with patch("app.utils.torchscript_utils.load_and_torchscript_model") as mock_load:
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_load.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_milvus():
    """Milvusのモックを提供"""
    with patch("pymilvus.connections.connect") as mock_connect:
        mock_connect.return_value = MagicMock()
        yield mock_connect


@pytest.fixture
def clean_registry():
    """テストごとにPrometheusメトリクスをクリーンアップ"""
    registry = CollectorRegistry()
    yield registry
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)


@pytest.fixture
def test_system(test_config, mock_model, mock_milvus):
    """テスト用のコンテキストシステムを提供"""
    system = ContextAwareSystem(test_config)
    return system


@pytest.fixture
def api_client(test_system):
    """テスト用のAPIクライアントを提供"""
    api_service = APIService(system=test_system)
    return TestClient(api_service.app)


@pytest.mark.usefixtures("mock_environment", "clean_registry")
def test_contextualize_endpoint(api_client):
    """contextualize エンドポイントのテスト"""
    # 認証なしでリクエスト（401エラーを期待）
    response = api_client.post("/v1/contextualize", json={
        "session_id": "test_session",
        "text": "Hello, DMHA!"
    })
    assert response.status_code == 401

    # 無効なトークンでリクエスト（401エラーを期待）
    response = api_client.post(
        "/v1/contextualize",
        json={"session_id": "test_session", "text": "Hello, DMHA!"},
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401


@pytest.mark.usefixtures("mock_environment", "clean_registry")
def test_system_initialization(test_system):
    """システム初期化のテスト"""
    assert test_system.redis is not None
    assert test_system.config is not None
    assert test_system.config.model_name == "distilbert-base-uncased"


@pytest.mark.usefixtures("mock_environment", "clean_registry")
def test_milvus_connection(test_system):
    """Milvus接続のテスト"""
    assert test_system.redis is not None
    assert test_system.config is not None
    # Milvusはスキップされるはず
    assert test_system.config.skip_milvus == True
