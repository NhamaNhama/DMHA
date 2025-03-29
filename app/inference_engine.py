import torch
import torch.nn as nn
from transformers import AutoTokenizer

class InferenceEngine(nn.Module):
    """
    Llama 2 7B Chatを想定。
    TorchScript or Rawモデル (self.base_model) を保持し、短期メモリも結合。
    """

    def __init__(self, scripted_model: torch.nn.Module, device: torch.device, model_name: str):
        super().__init__()
        self.base_model = scripted_model
        self.device = device
        self.base_model.to(self.device)
        self.base_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_size = 4096
        self.attention_router = nn.Linear(hidden_size, 8)
        self.temporal_encoder = nn.LSTM(hidden_size, 512, bidirectional=True)
        self.consistency_validator = self._build_validator(hidden_size)

        self.to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, memory_context=None):
        outputs = self.base_model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError("Model output missing last_hidden_state")

        dynamic_attention = self.attention_router(last_hidden_state)

        if memory_context is not None:
            memory_context = memory_context.to(self.device).float()
            temporal_encoded, _ = self.temporal_encoder(memory_context.unsqueeze(1))
            seq_len = last_hidden_state.shape[1]
            mem_len = temporal_encoded.shape[0]
            min_len = min(seq_len, mem_len)
            combined = torch.cat([
                last_hidden_state[:, :min_len, :],
                temporal_encoded[:min_len, 0, :].unsqueeze(0).expand(-1, min_len, -1)
            ], dim=-1)
            context_aware = combined
        else:
            context_aware = last_hidden_state

        avg_context = context_aware.mean(dim=1)
        consistency_score = self.consistency_validator(avg_context)

        logits = getattr(outputs, "logits", None)
        if logits is None:
            vocab_size = self.tokenizer.vocab_size
            fc = nn.Linear(context_aware.size(-1), vocab_size).to(self.device)
            logits = fc(context_aware)

        return {
            "logits": logits,
            "context_vectors": context_aware,
            "consistency_score": consistency_score
        }

    def _build_validator(self, hidden_size: int):
        return nn.Sequential(
            nn.Linear(hidden_size + 512*2, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )