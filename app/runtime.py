"""Inference backends. The stub is for tests only; Gemma receives audio directly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
import json
import math
from typing import Any

from .audio import WavAudio
from .settings import Settings


SYSTEM_PROMPT = """You are the reasoning component of the Sparkie robot. User input arrives as Italian audio.
Infer the user's intent directly from the audio. Answer concisely and naturally in Italian unless the user asks
for another language. Never claim a physical action succeeded before receiving execution feedback. You may call
only tools explicitly supplied in this request, and must not invent tool names or parameters. Ask for clarification
when a command is ambiguous or unsafe. Physical actions require stricter validation. Return only valid JSON matching
this schema: {request_id: string UUID, type: 'speech' or 'tool_call', response_text: string, tool_calls: [{name: string, arguments: object}]}."""


class InferenceRunner(ABC):
    device: str

    @abstractmethod
    def load(self) -> None:
        """Load the backend synchronously before the server becomes ready."""

    @abstractmethod
    def generate(self, audio: WavAudio, request_id: str, language: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> str:
        """Return the model's raw structured response."""


class StubRunner(InferenceRunner):
    """Deterministic test backend that never inspects or retains request audio."""

    device = "stub"

    def load(self) -> None:
        return None

    def generate(self, audio: WavAudio, request_id: str, language: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> str:
        return json.dumps(
            {
                "request_id": request_id,
                "type": "speech",
                "response_text": "Certo, come posso aiutarti?",
                "tool_calls": [],
            }
        )


class GemmaRunner(InferenceRunner):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.device
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Install the gemma extra before using MIND_RUNTIME=gemma.") from error
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(self.settings.precision)
        if dtype is None:
            raise RuntimeError("MIND_PRECISION must be bfloat16, float16, or float32.")
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.settings.device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.settings.device
        self.processor = AutoProcessor.from_pretrained(self.settings.model_id)
        self.model = Gemma3nForConditionalGeneration.from_pretrained(self.settings.model_id, **load_kwargs).eval()
        self.torch = torch
        self.device = str(getattr(self.model, "device", self.settings.device))

    def generate(self, audio: WavAudio, request_id: str, language: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> str:
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("Gemma runner has not been loaded.")
        waveform = self._decode_to_model_rate(audio)
        prompt = self.processor.apply_chat_template(
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": self._request_instruction(request_id, language, context, tools)},
                    ],
                },
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=prompt, audio=[waveform], return_tensors="pt", padding=True).to(
            self.model.device, dtype=self.model.dtype
        )
        input_length = inputs["input_ids"].shape[-1]
        generation_options: dict[str, Any] = {"max_new_tokens": self.settings.max_generated_tokens}
        if self.settings.temperature > 0:
            generation_options.update(do_sample=True, temperature=self.settings.temperature)
        else:
            generation_options["do_sample"] = False
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_options)
        return self.processor.decode(generated[0][input_length:], skip_special_tokens=True).strip()

    def _decode_to_model_rate(self, audio: WavAudio) -> Any:
        try:
            import numpy as np
            import soundfile as sf
            from scipy.signal import resample_poly
        except ImportError as error:
            raise RuntimeError("Install the gemma extra before using MIND_RUNTIME=gemma.") from error
        samples, sample_rate = sf.read(BytesIO(audio.content), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        target_rate = int(self.processor.feature_extractor.sampling_rate)
        if sample_rate != target_rate:
            divisor = math.gcd(sample_rate, target_rate)
            mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor).astype(np.float32)
        return mono

    @staticmethod
    def _request_instruction(request_id: str, language: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> str:
        return "\n".join(
            [
                f"Request ID: {request_id}",
                f"Language hint: {language}",
                f"Robot context: {json.dumps(context, ensure_ascii=False)}",
                f"Available tools: {json.dumps(tools, ensure_ascii=False)}",
                "Listen to the attached audio and return the required JSON only.",
            ]
        )


def build_runner(settings: Settings) -> InferenceRunner:
    return StubRunner() if settings.runtime == "stub" else GemmaRunner(settings)
