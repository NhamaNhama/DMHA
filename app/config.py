import os

class Config:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")

        self.redis_host = os.getenv("REDIS_HOST", "redis-master")
        self.milvus_host = os.getenv("MILVUS_HOST", "milvus-service")
        self.max_context_length = 4096
        self.memory_refresh_interval = 300
        self.cluster_threshold = 0.85

        self.jwt_secret = os.getenv("JWT_SECRET", "secret_key_for_demo")
        self.use_gpu = os.getenv("USE_GPU", "false").lower() == "true"

        self.cluster_method = os.getenv("CLUSTER_METHOD", "faiss")
        self.consistency_threshold = float(os.getenv("CONSISTENCY_THRESHOLD", "0.5"))

        self.torchscript_path = os.getenv("TORCHSCRIPT_PATH", "")

        # オンライン学習 (LoRA) 関連
        self.online_learning_enabled = os.getenv("ONLINE_LEARNING", "false").lower() == "true"
        self.lora_learning_rate = float(os.getenv("LORA_LEARNING_RATE", "1e-5"))
        self.lora_num_steps = int(os.getenv("LORA_NUM_STEPS", "10"))
        self.lora_peft_config = {
            "r": int(os.getenv("LORA_R", "8")),
            "lora_alpha": int(os.getenv("LORA_ALPHA", "32")),
            "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.05"))
        }