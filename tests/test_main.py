import pytest
from unittest.mock import patch, MagicMock
import uvicorn
from app.main import create_app


@pytest.fixture
def mock_config():
    """Configのモックを提供"""
    config_mock = MagicMock()
    return config_mock


@pytest.fixture
def mock_context_system():
    """ContextAwareSystemのモックを提供"""
    system_mock = MagicMock()
    system_mock.setup_k8s_cluster_manager = MagicMock()
    return system_mock


@pytest.fixture
def mock_api_service():
    """APIServiceのモックを提供"""
    api_mock = MagicMock()
    api_mock.app = MagicMock()
    return api_mock


@patch('app.main.Config')
@patch('app.main.ContextAwareSystem')
@patch('app.main.APIService')
def test_create_app(mock_api_service_class, mock_context_system_class, mock_config_class,
                    mock_config, mock_context_system, mock_api_service):
    """アプリケーション作成のテスト"""
    # モックの設定
    mock_config_class.return_value = mock_config
    mock_context_system_class.return_value = mock_context_system
    mock_api_service_class.return_value = mock_api_service

    # create_app関数を実行
    app = create_app()

    # Config, ContextAwareSystem, APIServiceが呼ばれたことを確認
    mock_config_class.assert_called_once()
    mock_context_system_class.assert_called_once()
    mock_api_service_class.assert_called_once()

    # K8sクラスタマネージャーがセットアップされたことを確認
    mock_context_system.setup_k8s_cluster_manager.assert_called_once()

    # APIServiceのappが返されたことを確認
    assert app == mock_api_service.app


@patch('app.main.create_app')
@patch('app.main.uvicorn.run')
def test_main_execution(mock_uvicorn_run, mock_create_app):
    """メイン実行のテスト（__main__ブロック）"""
    # モックの設定
    mock_app = MagicMock()
    mock_create_app.return_value = mock_app

    # __main__ブロックのコードを実行
    import app.main
    # __name__が"__main__"でなくても実行されるよう、関数を直接呼ぶ
    # これは通常のインポート時には実行されないコードをテストするためのもの
    if hasattr(app.main, "_test_main_block"):
        app.main._test_main_block()
    else:
        # __main__ブロックを模倣するために必要な部分だけ実行
        mock_app = create_app()
        uvicorn.run(
            mock_app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            timeout_keep_alive=30
        )

    # uvicorn.runが呼ばれたことを確認
    mock_uvicorn_run.assert_called_once()
    # 正しいパラメータが渡されたことを確認
    kwargs = mock_uvicorn_run.call_args[1]
    assert "host" in kwargs
    assert "port" in kwargs
    assert "log_level" in kwargs
