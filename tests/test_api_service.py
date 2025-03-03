from fastapi.testclient import TestClient
import os
import torch
from app.api_service import APIService
from app.context_system import ContextAwareSystem
from app.config import Config
from unittest.mock import patch, MagicMock
from app.utils.torchscript_utils import load_and_torchscript_model

# TorchScriptコンパイルをスキップするためのパッチ
def mock_load_model(*args, **kwargs):
    """テスト用のモックモデルを返す関数"""
    from transformers import AutoModel
    # 実際にモデルをロードするが、TorchScriptコンパイルはスキップ
    model_name = kwargs.get('model_name', args[0] if args else "distilbert-base-uncased")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model

# 実際のContextAwareSystemを使用するが、必要最小限の設定を行う
class TestContextSystem(ContextAwareSystem):
    def __init__(self):
        # テスト用の設定を作成
        test_config = Config()
        # ローカル環境用の設定
        test_config.redis_host = "localhost"
        test_config.milvus_host = "localhost"
        test_config.use_gpu = False
        
        # CIテスト環境ではオープンモデルを使用
        if os.getenv("SKIP_HF", "false").lower() == "true":
            # CIテスト環境では小さなオープンモデルを使用
            test_config.model_name = "distilbert-base-uncased"  # TorchScriptコンパイルに問題のないモデル
        else:
            # ローカルテスト環境では設定通りのモデルを使用
            pass
            
        # JWT認証のテスト用キー
        test_config.jwt_secret = "test_secret_key"
        
        # TorchScriptコンパイルをスキップするためのパッチを適用
        with patch("app.utils.torchscript_utils.load_and_torchscript_model", side_effect=mock_load_model):
            super().__init__(test_config)

# Milvusの接続だけをモック化（実際のMilvusサーバーがない環境でもテスト可能に）
@patch("pymilvus.connections.connect")
def test_contextualize_endpoint(mock_milvus_connect):
    """
    contextualize エンドポイントのテスト
    Milvus接続のみモック化し、それ以外は実際のコンポーネントを使用
    """
    # Milvusの接続をモック
    mock_milvus_connect.return_value = MagicMock()
    
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
    
    # 認証がない場合は401エラーを期待
    assert response.status_code == 401
    
    # 無効なトークンでリクエスト（401エラーを期待）
    response = client.post(
        "/v1/contextualize", 
        json={"session_id": "test_session", "text": "Hello, DMHA!"},
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401

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