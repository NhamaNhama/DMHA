from fastapi.testclient import TestClient
import os
import torch
from app.api_service import APIService
from app.context_system import ContextAwareSystem
from app.config import Config
from unittest.mock import patch, MagicMock
from app.utils.torchscript_utils import load_and_torchscript_model
from transformers import AutoModel

def mock_load_model(*args, **kwargs):
    """テスト用のモックモデルを返す関数"""
    model_name = kwargs.get('model_name', args[0] if args else "distilbert-base-uncased")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model

class TestContextSystem(ContextAwareSystem):
    def __init__(self):
        test_config = Config()
        test_config.redis_host = "localhost"
        test_config.milvus_host = "localhost"
        test_config.use_gpu = False
        test_config.model_name = "distilbert-base-uncased"
        test_config.jwt_secret = "test_secret_key"
        
        # Milvusの接続をモック化
        self.milvus_mock = MagicMock()
        with patch("pymilvus.connections.connect") as mock_connect:
            mock_connect.return_value = self.milvus_mock
            # TorchScriptコンパイルをスキップ
            with patch("app.utils.torchscript_utils.load_and_torchscript_model", side_effect=mock_load_model):
                super().__init__(test_config)

def test_contextualize_endpoint():
    """
    contextualize エンドポイントのテスト
    """
    # テスト用のシステムを作成
    test_system = TestContextSystem()
    
    # APIサービスを作成
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

def test_system_initialization():
    """システム初期化のテスト"""
    test_system = TestContextSystem()
    
    # システムが正しく初期化されたことを確認
    assert test_system.redis is not None
    assert test_system.config is not None
    assert test_system.config.model_name == "distilbert-base-uncased"

@patch("pymilvus.connections.connect")
def test_milvus_connection(mock_milvus_connect):
    """Milvus接続のテスト"""
    mock_milvus_connect.return_value = MagicMock()
    
    # テスト用のシステムを作成
    test_system = TestContextSystem()
    
    # Milvus接続が呼び出されたことを確認
    mock_milvus_connect.assert_called_once()
    
    # システムが正しく初期化されたことを確認
    assert test_system.redis is not None
    assert test_system.config is not None 