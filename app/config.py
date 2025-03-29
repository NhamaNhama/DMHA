from dataclasses import dataclass, field
import os
from typing import Dict, Any


@dataclass
class Config:
    # 基本設定
    model_name: str = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    redis_host: str = os.getenv("REDIS_HOST", "redis")
    milvus_host: str = os.getenv("MILVUS_HOST", "milvus")
    use_gpu: bool = os.getenv("USE_GPU", "true").lower() == "true"
    jwt_secret: str = os.getenv("JWT_SECRET", "your_jwt_secret")

    # メモリと一貫性設定
    consistency_threshold: float = float(
        os.getenv("CONSISTENCY_THRESHOLD", "0.7"))
    cluster_method: str = os.getenv("CLUSTER_METHOD", "faiss")

    # モデル設定
    torchscript_path: str = os.getenv("TORCHSCRIPT_PATH", None)

    # LoRA設定
    online_learning_enabled: bool = os.getenv(
        "ONLINE_LEARNING", "false").lower() == "true"
    lora_learning_rate: float = float(os.getenv("LORA_LEARNING_RATE", "5e-4"))
    lora_num_steps: int = int(os.getenv("LORA_NUM_STEPS", "50"))
    lora_peft_config: Dict[str, Any] = field(default_factory=lambda: {
        "r": int(os.getenv("LORA_R", "8")),
        "lora_alpha": int(os.getenv("LORA_ALPHA", "16")),
        "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.1")),
        "target_modules": ["query", "value"]
    })

    # テスト設定
    test_mode: bool = False

    @property
    def is_test_env(self) -> bool:
        """テスト環境かどうかを判定"""
        return self.test_mode or os.environ.get("CI", "false").lower() == "true"

    @property
    def skip_milvus(self) -> bool:
        """Milvus接続をスキップするかどうか"""
        return self.is_test_env or os.environ.get("SKIP_MILVUS", "false").lower() == "true"

    @property
    def skip_hf(self) -> bool:
        """HuggingFaceモデルを小さいものに置き換えるかどうか"""
        return self.is_test_env or os.environ.get("SKIP_HF", "false").lower() == "true"

    @property
    def effective_model_name(self) -> str:
        """実際に使用するモデル名"""
        return "prajjwal1/bert-tiny" if self.skip_hf else self.model_name
