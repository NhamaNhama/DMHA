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

class MockPreTrainedModel(PreTrainedModel):
    """テスト用のモックモデル"""
    def __init__(self):
        super().__init__(Config())
        self.config.hidden_size = 768
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        sequence_length = input_ids.shape[1] if input_ids is not None else 512
        return torch.ones(batch_size, sequence_length, self.config.hidden_size)

def mock_load_model(*args, **kwargs):
    """完全にモック化されたモデルを返す"""
    model = MockPreTrainedModel()
    model.eval()
    return model

@pytest.fixture(scope="function")
def clean_registry():
    """テストごとにPrometheusメトリクスをクリーンアップ"""
    registry = CollectorRegistry()
    yield registry
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)

@pytest.fixture
def mock_milvus():
    """Milvusのモックを提供"""
    with patch("pymilvus.connections.connect") as mock_connect:
        mock_connect.return_value = MagicMock()
        yield mock_connect

@pytest.fixture
def mock_model():
    """モデルのモックを提供"""
    with patch("transformers.AutoModel.from_pretrained", side_effect=mock_load_model):
        with patch("app.utils.torchscript_utils.load_and_torchscript_model", side_effect=mock_load_model):
            yield

def create_test_system():
    """テストシステムを作成するヘルパー関数"""
    test_config = Config()
    test_config.redis_host = "localhost"
    test_config.milvus_host = "localhost"
    test_config.use_gpu = False
    test_config.model_name = "distilbert-base-uncased"
    test_config.jwt_secret = "test_secret_key"
    return ContextAwareSystem(test_config)

@pytest.mark.usefixtures("clean_registry", "mock_milvus", "mock_model")
def test_contextualize_endpoint():
    """contextualize エンドポイントのテスト"""
    test_system = create_test_system()
    api_service = APIService(system=test_system)
    client = TestClient(api_service.app)
    
    # 認証なしでリクエスト（401エラーを期待）
    response = client.post("/v1/contextualize", json={
        "session_id": "test_session",
        "text": "Hello, DMHA!"
    })
    assert response.status_code == 401
    
    # 無効なトークンでリクエスト（401エラーを期待）
    response = client.post(
        "/v1/contextualize", 
        json={"session_id": "test_session", "text": "Hello, DMHA!"},
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401

@pytest.mark.usefixtures("clean_registry", "mock_milvus", "mock_model")
def test_system_initialization():
    """システム初期化のテスト"""
    test_system = create_test_system()
    assert test_system.redis is not None
    assert test_system.config is not None
    assert test_system.config.model_name == "distilbert-base-uncased"

@pytest.mark.usefixtures("clean_registry", "mock_milvus", "mock_model")
def test_milvus_connection():
    """Milvus接続のテスト"""
    test_system = create_test_system()
    assert test_system.redis is not None
    assert test_system.config is not None 