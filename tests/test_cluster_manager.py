import pytest
from unittest.mock import patch, MagicMock
import importlib.util

# kubernetesモジュールがインストールされているか確認
kubernetes_installed = importlib.util.find_spec("kubernetes") is not None

# モジュールがない場合はスキップするデコレータ
skip_if_no_kubernetes = pytest.mark.skipif(
    not kubernetes_installed,
    reason="kubernetes module is not installed"
)

# クラスのインポートをスキップ設定内に移動
if kubernetes_installed:
    from app.cluster_manager import ClusterManager


@pytest.fixture
def mock_k8s_config():
    """Kubernetesの設定をモック化"""
    with patch('kubernetes.config.load_incluster_config') as mock_config:
        yield mock_config


@pytest.fixture
def mock_k8s_api():
    """Kubernetes APIをモック化"""
    with patch('kubernetes.client.CoreV1Api') as mock_core_api:
        with patch('kubernetes.client.AppsV1Api') as mock_apps_api:
            mock_core_api_instance = MagicMock()
            mock_apps_api_instance = MagicMock()

            mock_core_api.return_value = mock_core_api_instance
            mock_apps_api.return_value = mock_apps_api_instance

            yield {
                'core': mock_core_api_instance,
                'apps': mock_apps_api_instance,
                'core_class': mock_core_api,
                'apps_class': mock_apps_api
            }


@pytest.fixture
def cluster_manager(mock_k8s_config, mock_k8s_api):
    """テスト用クラスタマネージャーを作成"""
    if not kubernetes_installed:
        pytest.skip("kubernetes module is not installed")
    manager = ClusterManager()
    return manager


@skip_if_no_kubernetes
def test_init(cluster_manager, mock_k8s_config, mock_k8s_api):
    """初期化のテスト"""
    # K8s設定が読み込まれていることを確認
    mock_k8s_config.assert_called_once()

    # APIクライアントが作成されていることを確認
    assert mock_k8s_api['core_class'].called
    assert mock_k8s_api['apps_class'].called

    # クラスタマネージャーのプロパティが正しく設定されていることを確認
    assert cluster_manager.api is not None
    assert cluster_manager.apps_api is not None


@skip_if_no_kubernetes
def test_scale_deployment(cluster_manager, mock_k8s_api):
    """デプロイメントスケーリングのテスト"""
    # テストパラメータ
    deployment_name = "test-deployment"
    replicas = 3
    namespace = "default"

    # 関数を実行
    cluster_manager.scale_deployment(deployment_name, replicas, namespace)

    # 正しいメソッドが呼び出されたことを確認
    mock_k8s_api['apps'].patch_namespaced_deployment_scale.assert_called_once()

    # 引数が正しいことを確認
    call_args = mock_k8s_api['apps'].patch_namespaced_deployment_scale.call_args[1]
    assert call_args['name'] == deployment_name
    assert call_args['namespace'] == namespace
    assert call_args['body'] == {"spec": {"replicas": replicas}}


@skip_if_no_kubernetes
def test_scale_deployment_custom_namespace(cluster_manager, mock_k8s_api):
    """カスタム名前空間でのデプロイメントスケーリングのテスト"""
    # テストパラメータ
    deployment_name = "test-deployment"
    replicas = 5
    namespace = "production"

    # 関数を実行
    cluster_manager.scale_deployment(deployment_name, replicas, namespace)

    # 正しいメソッドが呼び出されたことを確認
    mock_k8s_api['apps'].patch_namespaced_deployment_scale.assert_called_once()

    # 引数が正しいことを確認
    call_args = mock_k8s_api['apps'].patch_namespaced_deployment_scale.call_args[1]
    assert call_args['name'] == deployment_name
    assert call_args['namespace'] == namespace
    assert call_args['body'] == {"spec": {"replicas": replicas}}
