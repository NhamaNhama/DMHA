import torch
from transformers import AutoModel

def load_and_torchscript_model(model_name: str, device: torch.device, script_path: str = "") -> torch.nn.Module:
    if script_path:
        try:
            model_ts = torch.jit.load(script_path, map_location=device)
            model_ts.eval()
            print("[INFO] Loaded pre-compiled TorchScript model.")
            return model_ts
        except Exception as e:
            print(f"[WARNING] Failed to load pre-compiled TorchScript: {e}")

    base_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if (torch.cuda.is_available() and device.type == "cuda") else torch.float32
    )
    base_model.eval().to(device)

    try:
        scripted = torch.jit.script(base_model)
        optimized = torch.jit.optimize_for_inference(scripted)
        frozen = torch.jit.freeze(optimized)
        print("[INFO] TorchScript compilation succeeded.")
        return frozen
    except Exception as e:
        print(f"[WARNING] TorchScript compilation failed. Fallback to raw model: {e}")
        return base_model