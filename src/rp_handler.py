import os
import io
import time
import base64
import traceback
from typing import Any, Dict

import requests
from PIL import Image
import torch
import runpod

# ---- MoE-LLaVA imports (официальные) ----
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.mm_utils import (
    tokenizer_image_token,
    KeywordsStoppingCriteria,
    get_model_name_from_path,
)
from moellava.utils import disable_torch_init
# -----------------------------------------

# ---------- ENV / Defaults ----------
MODEL_ID = os.environ.get("MODEL_ID", "LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384")
LOAD_4BIT = os.environ.get("LOAD_4BIT", "false").lower() == "true"
LOAD_8BIT = os.environ.get("LOAD_8BIT", "false").lower() == "true"
PRECISION = os.environ.get("PRECISION", "fp16").lower()  # fp16|bf16
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")
TOKENIZERS_PARALLELISM = os.environ.get("TOKENIZERS_PARALLELISM", "false")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
disable_torch_init()
# ------------------------------------

def _select_dtype() -> torch.dtype:
    if PRECISION == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def _load_image_any(x: str) -> Image.Image:
    # принимает base64 или URL
    try:
        raw = base64.b64decode(x, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        if x.startswith("http://") or x.startswith("https://"):
            r = requests.get(x, timeout=30)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    raise ValueError("Provide 'input_image' as base64 or http(s) URL.")

def _infer_conv_mode(model_id: str) -> str:
    mid = model_id.lower()
    if "qwen" in mid: return "qwen"
    if "stablelm" in mid: return "stablelm"
    # дефолт для Phi2/прочих
    return "phi"

class MoeLlavaService:
    _loaded = False

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv_mode = _infer_conv_mode(MODEL_ID)

    def load(self):
        if self._loaded:
            return
        t0 = time.time()
        if HF_TOKEN:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

        model_name = get_model_name_from_path(MODEL_ID)
        self.tokenizer, self.model, processor, self.context_len = load_pretrained_model(
            MODEL_ID, None, model_name, LOAD_8BIT, LOAD_4BIT, device=self.device
        )
        self.image_processor = processor["image"]
        try:
            self.model.to(dtype=_select_dtype())
        except Exception:
            pass
        self.model.eval()
        self._loaded = True
        print(f"[MoE-LLaVA] Loaded {MODEL_ID} in {time.time()-t0:.1f}s | 4bit={LOAD_4BIT} 8bit={LOAD_8BIT} conv={self.conv_mode}")

    @torch.inference_mode()
    def generate(self, prompt: str, image: Image.Image, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            self.load()

        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles

        # один шаг диалога: всегда добавляем image-токен
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(roles[0], inp)
        conv.append_message(roles[1], None)
        text_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"].to(self.model.device, dtype=_select_dtype())

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        temperature = float(params.get("temperature", 0.2))
        top_p = float(params.get("top_p", 0.95))
        max_new_tokens = int(params.get("max_new_tokens", MAX_NEW_TOKENS))
        do_sample = temperature > 0

        t0 = time.time()
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
        )
        t_gen = time.time() - t0

        # отрезаем префикс промпта
        gen_text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        # на случай, если в конце зацепился сепаратор
        if gen_text.endswith(stop_str):
            gen_text = gen_text[:-len(stop_str)].strip()

        return {
            "text": gen_text,
            "generation_time_sec": round(t_gen, 3),
            "model": MODEL_ID,
            "params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens
            }
        }

SERVICE = MoeLlavaService()
SERVICE.load()  # прогрев на холодном старте

# -------- Валидация входа (совместимо с replicate camenduru/moe-llava) --------
def _parse_event(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict) or "input" not in event:
        raise ValueError("Payload must contain 'input' object.")
    ip = event["input"] or {}
    img = ip.get("input_image")
    txt = ip.get("input_text", "")
    if not img:
        raise ValueError("'input_image' is required (base64 or URL).")
    if not isinstance(txt, str) or not txt:
        txt = "Describe the image."
    params = {
        "temperature": ip.get("temperature", 0.2),
        "top_p": ip.get("top_p", 0.95),
        "max_new_tokens": ip.get("max_new_tokens", MAX_NEW_TOKENS),
    }
    return {"image": img, "prompt": txt, "params": params}
# -----------------------------------------------------------------------------

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Вход (как у replicate):
    {
      "input": {
        "input_image": "<base64> | https://...",
        "input_text": "What is unusual about this image?",
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 256
      }
    }
    """
    try:
        parsed = _parse_event(event)
        image = _load_image_any(parsed["image"])
        out = SERVICE.generate(parsed["prompt"], image, parsed["params"])
        # Возвращаем так, чтобы 'output' сразу содержал строку (как в replicate)
        return {"output": out["text"], "meta": out}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print("[handler][ERROR]", err)
        print(traceback.format_exc())
        return {"error": err}

if __name__ == "__main__":
    # запуск RunPod serverless-воркера
    runpod.serverless.start({"handler": handler})
