from fastapi.testclient import TestClient
from app.api_service import APIService
from app.context_system import ContextAwareSystem
from app.config import Config

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

def test_contextualize_endpoint():
    # モックシステムで APIService を初期化し、テストクライアントを生成
    mock_system = MockSystem()
    api_service = APIService(system=mock_system)
    client = TestClient(api_service.app)

    # テスト用のリクエスト
    # 実際にはトークンが必要な場合があるので、その場合はヘッダに Bearer トークンを付与してください
    response = client.post("/v1/contextualize", json={
        "session_id": "test_session",
        "text": "Hello, DMHA!"
    })

    # 予想されるステータスコードやレスポンス構造を検証
    assert response.status_code == 401, "未認証の場合の応答コードを想定"

    # 必要に応じてトークンを発行し、認証付きで再テストする例：
    # token_response = client.post("/auth/token", params={"username": "testuser"})
    # token = token_response.json()["access_token"]
    # authenticated_response = client.post(
    #     "/v1/contextualize",
    #     json={"session_id": "test_session", "text": "Hello with token!"},
    #     headers={"Authorization": f"Bearer {token}"}
    # )
    # assert authenticated_response.status_code == 200 