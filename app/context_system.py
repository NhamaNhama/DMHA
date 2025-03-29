import logging
import torch
from prometheus_client import start_http_server, Summary, Gauge, Counter
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from redis import Redis
import os

from .config import Config
from .meta_cognition import MetaCognitionModule
from .symbolic_interface import NeuralSymbolicInterface
from .memory_manager import MemoryManager
from .inference_engine import InferenceEngine
from .utils.torchscript_utils import load_and_torchscript_model
from .online_learning import OnlineLearner


class ContextAwareSystem:
    """
    DMHA中枢: メタ認知制御, シンボリック推論, オンライン学習(LoRA)などを統合。
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.metrics = self._setup_metrics()

        # Redis接続
        self.redis = Redis(host=config.redis_host, decode_responses=True)

        # Milvus接続（必要な場合のみ）
        self.milvus = None
        if not config.skip_milvus:
            try:
                self.milvus = connections.connect(
                    alias="default",
                    host=config.milvus_host,
                    port="19530"
                )
            except Exception as e:
                self.logger.warning(f"Milvus接続エラー: {str(e)}")
        else:
            self.logger.info("Milvus接続をスキップします")

        # デバイス設定
        device = torch.device("cuda" if (
            config.use_gpu and torch.cuda.is_available()) else "cpu")
        self.logger.info(f"Using device: {device}")

        # モデルロード
        self.model = load_and_torchscript_model(
            model_name=config.effective_model_name,
            device=device,
            script_path=config.torchscript_path
        )

        # Milvusコレクション初期化（Milvus接続時のみ）
        if self.milvus and not config.is_test_env:
            self._init_milvus_collections()

        self.meta_cognition = MetaCognitionModule(
            threshold=config.consistency_threshold)
        self.ns_interface = NeuralSymbolicInterface()

        self._inference_engine = None
        self.device = device

        # オンライン学習 (LoRA) - テストモードでなく、有効な場合のみ
        self.online_learner = None
        if config.online_learning_enabled and not config.is_test_env:
            self.online_learner = OnlineLearner(
                base_model=self.model,
                device=device,
                tokenizer_name=config.model_name,
                lr=config.lora_learning_rate,
                num_steps=config.lora_num_steps,
                lora_config=config.lora_peft_config
            )

    def _setup_logger(self):
        logger = logging.getLogger("ContextEngine")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{"timestamp":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}'
        ))
        logger.addHandler(handler)
        return logger

    def _setup_metrics(self):
        metrics = {
            'request_latency': Summary('request_processing_seconds', 'Time spent processing request'),
            'memory_usage': Gauge('memory_usage_bytes', 'Current memory usage'),
            'conflict_score': Gauge('context_conflict_score', 'Consistency conflict score'),
            'inference_count': Counter('inference_requests_total', 'Total number of inference requests'),
            'inference_latency': Summary('inference_processing_seconds', 'Time spent in model inference')
        }
        # テスト環境ではPrometheusサーバーを起動しない
        if not self.config.is_test_env and os.environ.get("PYTEST_CURRENT_TEST", "") == "":
            try:
                start_http_server(8000)
            except OSError:
                self.logger.warning(
                    "ポート8000が既に使用されています。Prometheusメトリクスサーバーは起動しません。")
        return metrics

    def _init_milvus_collections(self):
        collection_params = {
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "embedding", "type": "float_vector", "dim": 4096},
                {"name": "timestamp", "type": "int64"},
                {"name": "entity_tags", "type": "json"},
                {"name": "semantic_cluster", "type": "int16"}
            ],
            "segment_row_limit": 4096,
            "auto_id": False
        }
        if not self.milvus.has_collection("long_term_memory"):
            self.milvus.create_collection(
                "long_term_memory", collection_params)

    def get_inference_engine(self) -> InferenceEngine:
        if self._inference_engine is None:
            self._inference_engine = InferenceEngine(
                scripted_model=self.model,
                device=self.device,
                model_name=self.config.model_name
            )
        return self._inference_engine

    def setup_k8s_cluster_manager(self):
        try:
            from kubernetes import client, config as k8s_config
            k8s_config.load_incluster_config()
            self.k8s_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
        except Exception as e:
            self.logger.error(f"K8s config load failed: {str(e)}")

    def scale_deployment(self, deployment_name: str, replicas: int):
        try:
            body = {"spec": {"replicas": replicas}}
            self.apps_api.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace="default",
                body=body
            )
        except Exception as e:
            self.logger.error(f"Failed to scale deployment: {str(e)}")

    def connect_to_milvus(self):
        # 実接続は SKIP_MILVUS が "true" でない場合にのみ実行
        if os.environ.get("SKIP_MILVUS", "false").lower() == "true":
            print("Skipping real Milvus connection due to SKIP_MILVUS env.")
        else:
            print("Connecting to Milvus...")
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port="19530"
            )

    def trigger_online_learning(self, training_data):
        """
        training_data: オンライン学習に必要となるデータ（例: 会話履歴や追加アノテーション）など。
        """
        if not self.online_learner:
            self.logger.warning(
                "オンライン学習が有効ではありません。config.online_learning_enabled を確認してください。")
            return

        # データのバリデーション
        if not training_data or len(training_data) == 0:
            self.logger.error("学習データが空、または無効です。")
            return

        try:
            # 実際のオンライン学習を実施
            learning_result = self.online_learner.train(training_data)
            self.logger.info(f"オンライン学習が成功しました。結果: {learning_result}")
        except Exception as e:
            self.logger.error(f"オンライン学習中にエラーが発生しました: {str(e)}")
