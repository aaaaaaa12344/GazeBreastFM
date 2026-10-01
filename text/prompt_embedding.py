from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable
from pathlib import Path

import torch


def _deterministic_prompt_vector(prompt: str, text_dim: int) -> torch.Tensor:
    if text_dim <= 0:
        raise ValueError("text_dim must be positive.")

    values: list[float] = []
    seed = prompt.encode("utf-8")
    counter = 0
    while len(values) < text_dim:
        digest = hashlib.sha256(seed + counter.to_bytes(4, byteorder="little")).digest()
        for byte_value in digest:
            values.append((float(byte_value) / 255.0) * 2.0 - 1.0)
            if len(values) == text_dim:
                break
        counter += 1
    return torch.tensor(values, dtype=torch.float32)


def _adapt_embedding_dim(
    embedding: torch.Tensor,
    expected_dim: int,
    prompt: str,
) -> torch.Tensor:
    current_dim = int(embedding.shape[0])
    if current_dim == expected_dim:
        return embedding.to(dtype=torch.float32)

    if current_dim < expected_dim:
        warnings.warn(
            f"Prompt embedding dim for '{prompt}' is {current_dim}, padded to {expected_dim}.",
            stacklevel=2,
        )
        padded = torch.zeros((expected_dim,), dtype=torch.float32)
        padded[:current_dim] = embedding.to(dtype=torch.float32)
        return padded

    warnings.warn(
        f"Prompt embedding dim for '{prompt}' is {current_dim}, truncated to {expected_dim}.",
        stacklevel=2,
    )
    return embedding[:expected_dim].to(dtype=torch.float32)


class PromptEmbeddingCacheEncoder:
    def __init__(
        self,
        cache_path: str | Path,
        text_dim: int,
        *,
        allow_missing_fallback: bool = True,
        required_prompt_path: str | Path | None = None,
    ) -> None:
        self.cache_path = Path(cache_path).expanduser().resolve()
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Prompt embedding cache does not exist: {self.cache_path}")

        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        raw_embeddings = payload.get("prompt_embeddings", {})
        if not isinstance(raw_embeddings, dict):
            raise ValueError(
                f"prompt_embeddings must be a mapping in {self.cache_path}."
            )

        self.text_backend = "prompt_embedding_cache"
        self.text_dim = int(text_dim)
        self.allow_missing_fallback = bool(allow_missing_fallback)
        self.prompt_embeddings: dict[str, torch.Tensor] = {}
        for prompt, raw_vector in raw_embeddings.items():
            vector = torch.as_tensor(raw_vector, dtype=torch.float32).flatten()
            self.prompt_embeddings[str(prompt)] = _adapt_embedding_dim(
                vector,
                expected_dim=self.text_dim,
                prompt=str(prompt),
            )

        self.coverage_summary = {
            "required_prompt_count": 0,
            "resolved_prompt_count": 0,
            "missing_prompt_count": 0,
        }
        if required_prompt_path is not None:
            required_prompts = _load_required_prompts(required_prompt_path)
            self.validate_prompt_coverage(required_prompts)

    def validate_prompt_coverage(self, prompts: Iterable[str]) -> None:
        """Require every frozen Effective Report prompt key in the supplied authority."""
        normalized = {str(prompt).strip() for prompt in prompts if str(prompt).strip()}
        missing = sorted(normalized - set(self.prompt_embeddings))
        self.coverage_summary = {
            "required_prompt_count": len(normalized),
            "resolved_prompt_count": len(normalized) - len(missing),
            "missing_prompt_count": len(missing),
        }
        if missing:
            raise KeyError(
                "Prompt embedding authority coverage failure: "
                f"{len(missing)} required prompt key(s) are missing from {self.cache_path}."
            )

    def encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, list[str]]:
        embeddings: list[torch.Tensor] = []
        warning_messages: list[str] = []
        for prompt in prompts:
            cached = self.prompt_embeddings.get(prompt)
            if cached is None:
                warning_message = f"prompt_embedding_cache_miss:{prompt}"
                if not self.allow_missing_fallback:
                    raise KeyError(
                        f"Prompt embedding authority missing key in formal mode: {prompt!r}"
                    )
                warnings.warn(warning_message, stacklevel=2)
                warning_messages.append(warning_message)
                cached = _deterministic_prompt_vector(prompt, text_dim=self.text_dim)
            embeddings.append(cached)
        return torch.stack(embeddings, dim=0), warning_messages


def _load_required_prompts(path: str | Path) -> list[str]:
    required_path = Path(path).expanduser().resolve()
    if not required_path.is_file():
        raise FileNotFoundError(f"Required Effective Report prompt authority does not exist: {required_path}")
    prompts: list[str] = []
    with required_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid prompt authority JSONL at {required_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Prompt authority row at {required_path}:{line_number} must be an object."
                )
            prompt = str(value.get("text_prompt") or value.get("effective_report_text") or "").strip()
            if not prompt:
                raise ValueError(
                    f"Prompt authority row at {required_path}:{line_number} lacks text_prompt/effective_report_text."
                )
            prompts.append(prompt)
    return prompts
