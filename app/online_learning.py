import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import List, Tuple
import traceback

class OnlineLearner:
    """
    LoRAによるオンライン微調整 & 学習後のTorchScript再適用。
    """

    def __init__(
        self,
        base_model: nn.Module,
        device: torch.device,
        tokenizer_name: str,
        lr: float = 1e-5,
        num_steps: int = 10,
        lora_config: dict = None
    ):
        self.base_model = base_model
        self.device = device
        self.lr = lr
        self.num_steps = num_steps

        if lora_config is None:
            lora_config = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.05}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from peft import LoraConfig, get_peft_model
        self.lora_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=["q_proj", "v_proj"]
        )

        self.peft_model = get_peft_model(self.base_model, self.lora_config)
        self.peft_model.train()
        self.peft_model.to(self.device)

    def apply_online_learning(self, samples: List[Tuple[str, str]]):
        if not samples:
            return

        optimizer = optim.AdamW(self.peft_model.parameters(), lr=self.lr)

        for step in range(self.num_steps):
            total_loss = 0.0
            for (inp, tgt) in samples:
                prompt_text = f"{inp}\n{tgt}"
                encoding = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                optimizer.zero_grad()
                outputs = self.peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # ここでstep毎のログをprintするなど
            # print(f"LoRA step {step+1}/{self.num_steps}, loss={total_loss:.4f}")

        self.peft_model.eval()

        # base_modelにも更新が反映
        self.base_model = self.peft_model

    def rescript_model(self):
        """
        学習後にTorchScriptを再適用して推論に反映。
        """
        try:
            scripted = torch.jit.script(self.peft_model)
            optimized = torch.jit.optimize_for_inference(scripted)
            frozen = torch.jit.freeze(optimized)
            print("[INFO] Re-TorchScript succeeded after LoRA training.")
            self.peft_model = frozen
            self.base_model = frozen
        except Exception as e:
            print("[WARNING] Re-TorchScript failed after LoRA training:")
            traceback.print_exc()