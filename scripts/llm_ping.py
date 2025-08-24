from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path


def _read_env_file(dotenv_path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
                if not m:
                    continue
                key, val = m.group(1), m.group(2).strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                env[key] = val
    except FileNotFoundError:
        pass
    return env


def _normalize_json_text(text: str) -> str:
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else text


def _parse_json_tolerant(text: str):
    cleaned = _normalize_json_text(text)
    try:
        return json.loads(cleaned), True
    except Exception:
        t = re.sub(r"(\w+)\s*:\s*", r'"\\1": ', cleaned)
        try:
            return json.loads(t), True
        except Exception:
            return None, False


def main() -> None:
    # Ensure project root is importable
    if __package__ in (None, ""):
        sys.path.append(str(Path(__file__).resolve().parents[1]))

    project_root = Path(__file__).resolve().parents[1]

    # Load .env into environment for backend/model if not already set
    env_from_file = _read_env_file(str(project_root / ".env"))
    for k in ("LLM_BACKEND", "MODEL_NAME"):
        if k in env_from_file and not os.getenv(k):
            os.environ[k] = env_from_file[k]

    backend = os.getenv("LLM_BACKEND", "<unset>")
    model = os.getenv("MODEL_NAME", "<unset>")

    from recjudge.llm import call_llm  # type: ignore

    prompt = (
        "请严格输出 JSON：{\"ok\": true}\n"
        "要求：不要任何额外文字、解释或 Markdown，禁止返回除 JSON 以外的字符。"
    )

    started = time.perf_counter()
    raw = ""
    parsed_ok = False
    exc_type = None
    exc_msg = None
    try:
        raw = call_llm(prompt, temperature=0.0, max_tokens=16, model=model if model != "<unset>" else None)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _, parsed_ok = _parse_json_tolerant(raw)
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        exc_type = e.__class__.__name__
        exc_msg = str(e)

    snippet = (raw or "")[:100].replace("\n", " ")
    print(f"backend={backend}")
    print(f"model={model}")
    print(f"elapsed_ms={elapsed_ms}")
    print(f"snippet={snippet}")
    print(f"parsed_ok={parsed_ok}")
    if exc_type is not None:
        print(f"exception={exc_type}: {exc_msg}")


if __name__ == "__main__":
    main()


