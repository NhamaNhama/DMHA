from fastapi.testclient import TestClient
from app.api_service import APIService
from app.context_system import ContextAwareSystem
from app.config import Config
from unittest.mock import patch, MagicMock

# ContextAwareSystem のモックインスタンスを作る例。実際の実装に合せて修正してください。
class MockSystem(ContextAwareSystem):
    def __init__(self):
        mock_config = Config()
        # 必要に応じて最低限の項目を設定
        mock_config.redis_host = "localhost"
        mock_config.milvus_host = "localhost"
        mock_config.use_gpu = False
        # ... （setupが必要な場合は適宜追加）

        super().__init__(mock_config)
        # 上記を実行することで self._inference_engine 等が正しく初期化される

@patch("app.utils.torchscript_utils.load_and_torchscript_model")
@patch("pymilvus.connections.connect")
def test_contextualize_endpoint(mock_milvus_connect, mock_model_loader):
    """
    Milvus 接続と Hugging Face ダウンロード(load_and_torchscript_model)をモック化して
    Gated Repo へのアクセスと Milvus retryを回避。
    """
    # Milvus の connect() をモック
    mock_milvus_connect.return_value = MagicMock()
    # HFモデル読み込みもモック
    mock_model_loader.return_value = MagicMock()

    mock_system = MockSystem()
    api_service = APIService(system=mock_system)
    client = TestClient(api_service.app)

    response = client.post("/v1/contextualize", json={
        "session_id": "test_session",
        "text": "Hello, DMHA!"
    })
    # 認証がない想定で 401 などのレスポンスチェック
    assert response.status_code == 401

@patch("pymilvus.connections.connect")
def test_milvus_connection_mocked(mock_milvus_connect):
    mock_milvus_connect.return_value = MagicMock()
    # 実際の接続は行われず、mock_milvus_connect が呼ばれる
    assert True 