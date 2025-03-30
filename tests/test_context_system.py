import pytest
from unittest.mock import patch, MagicMock
import torch
import os
from app.context_system import ContextAwareSystem
from app.config import Config
from app.inference_engine import InferenceEngine
from prometheus_client import REGISTRY, CollectorRegistry


@pytest.fixture(scope="function", autouse=True)
def clean_prometheus_registry():
    """テストごとにPrometheusメトリクスをクリーンアップ"""
    # テスト前にレジストリをクリア
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    yield
    # テスト後にもクリア
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


@pytest.fixture
def mock_environment(monkeypatch):
    """テスト環境用の設定を一時的に適用するfixture"""
    monkeypatch.setenv("SKIP_MILVUS", "true")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    monkeypatch.setenv("CI", "true")  # CI環境としてマーク
    yield


@pytest.fixture
def mock_model():
    """モデルのモックを提供"""
    model_mock = MagicMock()
    model_mock.config.hidden_size = 768
    return model_mock


@pytest.fixture
def test_config():
    """テスト用の設定を提供"""
    config = Config()
    config.redis_host = "localhost"
    config.milvus_host = "localhost"
    config.use_gpu = False
    config.model_name = "distilbert-base-uncased"
    config.torchscript_path = None
    config.online_learning_enabled = False
    config.test_mode = True
    # test_modeをTrueにすることで、skip_milvusプロパティもTrueを返す
    return config


@pytest.fixture
def context_system(test_config, mock_model, mock_environment):
    """テスト用のコンテキストシステムを提供"""
    with patch('app.utils.torchscript_utils.load_and_torchscript_model', return_value=mock_model):
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value = MagicMock()
            with patch('pymilvus.connections.connect') as mock_connect:
                with patch('prometheus_client.start_http_server'):  # Prometheusサーバーをモック
                    # InferenceEngineをモック
                    with patch.object(InferenceEngine, '__init__', return_value=None):
                        # メトリクス設定をモック
                        with patch.object(ContextAwareSystem, '_setup_metrics', return_value={
                            'request_latency': MagicMock(),
                            'memory_usage': MagicMock(),
                            'conflict_score': MagicMock(),
                            'inference_count': MagicMock(),
                            'inference_latency': MagicMock()
                        }):
                            system = ContextAwareSystem(config=test_config)
                            system._inference_engine = MagicMock()  # 推論エンジンをモック
                            yield system


@pytest.mark.usefixtures("mock_environment")
def test_init(context_system, test_config):
    """初期化のテスト"""
    assert context_system.config == test_config
    assert context_system.logger is not None
    assert context_system.redis is not None
    assert context_system.model is not None
    assert context_system.meta_cognition is not None
    assert context_system.ns_interface is not None
    assert context_system.online_learner is None  # オンライン学習は無効化している


@pytest.mark.usefixtures("mock_environment")
def test_get_inference_engine(context_system):
    """推論エンジン取得のテスト"""
    engine = context_system.get_inference_engine()
    assert engine is not None
    # 同じインスタンスが返されることを確認
    engine2 = context_system.get_inference_engine()
    assert engine is engine2


@pytest.mark.usefixtures("mock_environment")
def test_setup_metrics(context_system):
    """メトリクス設定のテスト"""
    metrics = context_system.metrics
    assert 'request_latency' in metrics
    assert 'memory_usage' in metrics
    assert 'conflict_score' in metrics
    assert 'inference_count' in metrics
    assert 'inference_latency' in metrics


@pytest.mark.skip(reason="kubernetesモジュールがインストールされていないためスキップ")
@pytest.mark.usefixtures("mock_environment")
@patch('kubernetes.client.CoreV1Api')
@patch('kubernetes.client.AppsV1Api')
@patch('kubernetes.config.load_incluster_config')
def test_setup_k8s_cluster_manager(mock_load_config, mock_apps_api, mock_core_api, context_system):
    """K8sクラスタマネージャーセットアップのテスト"""
    # モックを返すように設定
    mock_core_api.return_value = MagicMock()
    mock_apps_api.return_value = MagicMock()

    context_system.setup_k8s_cluster_manager()
    mock_load_config.assert_called_once()


@pytest.mark.usefixtures("mock_environment")
def test_scale_deployment(context_system):
    """デプロイメントスケーリングのテスト"""
    # K8sクライアント設定
    context_system.apps_api = MagicMock()

    # スケーリング実行
    context_system.scale_deployment("test-deployment", 3)

    # 正しいメソッドが呼び出されたことを確認
    context_system.apps_api.patch_namespaced_deployment_scale.assert_called_once()
    args, kwargs = context_system.apps_api.patch_namespaced_deployment_scale.call_args
    assert kwargs["name"] == "test-deployment"
    assert kwargs["namespace"] == "default"
    assert kwargs["body"] == {"spec": {"replicas": 3}}


@pytest.mark.usefixtures("mock_environment")
def test_connect_to_milvus(context_system):
    """Milvus接続のテスト"""
    with patch('pymilvus.connections.connect') as mock_connect:
        # SKIP_MILVUS環境変数があるので接続はスキップされるはず
        context_system.connect_to_milvus()
        mock_connect.assert_not_called()


@pytest.mark.usefixtures("mock_environment")
def test_online_learning_disabled(context_system):
    """オンライン学習無効時のテスト"""
    # オンライン学習は無効化されている
    training_data = [{"input": "test", "output": "test"}]
    context_system.trigger_online_learning(training_data)

    # オンライン学習モジュールが作成されていないことを確認
    assert context_system.online_learner is None
