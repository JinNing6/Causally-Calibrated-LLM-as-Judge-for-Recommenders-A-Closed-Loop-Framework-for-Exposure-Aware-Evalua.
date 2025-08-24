from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_DOTENV_LOADED = False


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required but not set")
    return value


def _load_env_from_dotenv_if_needed() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        project_root = Path(__file__).resolve().parents[1]
        dotenv_path = project_root / ".env"
        if dotenv_path.exists():
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        # Silently ignore .env loading issues; explicit env vars still work
        pass
    finally:
        _DOTENV_LOADED = True


def call_llm(prompt: str, *, temperature: float = 0.2, max_tokens: int = 256, model: Optional[str] = None) -> str:
    """Call a real LLM backend and return the raw text content.

    Environment variables:
      - LLM_BACKEND: qwen|openai (default: qwen)
      - MODEL_NAME: optional model name; defaults to backend-specific default

    Qwen (DashScope, OpenAI-compatible) expects:
      - QWEN_API_KEY or DASHSCOPE_API_KEY
      - optional QWEN_BASE_URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)

    Tongyi (DashScope native) expects:
      - QWEN_API_KEY or DASHSCOPE_API_KEY or TONGYI_API_KEY
      - optional DASHSCOPE_API_URL (default: https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation)

    OpenAI expects:
      - OPENAI_API_KEY
      - optional OPENAI_BASE_URL (rarely used)
    """
    _load_env_from_dotenv_if_needed()
    backend = (os.getenv("LLM_BACKEND") or "qwen").lower()

    if backend in {"tongyi", "dashscope-native"}:
        model_name = model or os.getenv("MODEL_NAME") or "qwen-plus"
        api_key = (
            os.getenv("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("TONGYI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Tongyi backend requires QWEN_API_KEY or DASHSCOPE_API_KEY (or TONGYI_API_KEY) to be set"
            )

        # Prefer Python SDK if available
        try:
            import json as _json  # local alias to avoid shadowing
            try:
                import dashscope  # type: ignore
                from dashscope import Generation  # type: ignore

                dashscope.api_key = api_key  # type: ignore[attr-defined]
                resp = Generation.call(  # type: ignore[attr-defined]
                    model=model_name,
                    input={"messages": [{"role": "user", "content": prompt}]},
                    parameters={
                        "result_format": "message",
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                    },
                )
                # dashscope Response has .output dict
                output = getattr(resp, "output", None)
                if isinstance(output, dict):
                    if "choices" in output and output["choices"]:
                        msg = output["choices"][0].get("message") or {}
                        content = str(msg.get("content", ""))
                        if content:
                            return content
                    text = output.get("text")
                    if isinstance(text, str) and text:
                        return text
                # Fallback to stringified response
                try:
                    return _json.dumps(output, ensure_ascii=False)
                except Exception:
                    return str(output)
            except Exception:
                # Fallback to HTTP if SDK missing or fails
                import json
                import urllib.request

                url = (
                    os.getenv("DASHSCOPE_API_URL")
                    or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
                )
                payload = {
                    "model": model_name,
                    "input": {"messages": [{"role": "user", "content": prompt}]},
                    "parameters": {
                        "result_format": "message",
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                    },
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8", errors="ignore"))
                output = data.get("output") or data
                if isinstance(output, dict):
                    if "choices" in output and output["choices"]:
                        msg = output["choices"][0].get("message") or {}
                        content = str(msg.get("content", ""))
                        if content:
                            return content
                    text = output.get("text")
                    if isinstance(text, str) and text:
                        return text
                return str(output)
        except Exception as e:  # pragma: no cover - robust fallback
            raise RuntimeError(f"Tongyi request failed: {e}")

    if backend in {"qwen", "dashscope"}:
        # Prefer explicit model, otherwise env or sensible default for Qwen
        model_name = model or os.getenv("MODEL_NAME") or "qwen-plus-latest"

        # DashScope provides an OpenAI-compatible endpoint; we use the official OpenAI SDK
        api_key = (
            os.getenv("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")  # allow reuse if user exported only this
        )
        if not api_key:
            raise RuntimeError("Qwen backend requires QWEN_API_KEY or DASHSCOPE_API_KEY (or OPENAI_API_KEY) to be set")

        base_url = os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    if backend == "openai":
        model_name = model or os.getenv("MODEL_NAME") or "gpt-4o-mini"
        api_key = _require_env("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")  # optional

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    raise RuntimeError(f"Unsupported LLM_BACKEND: {backend}")


