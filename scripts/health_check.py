from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------- utils ----------------------


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_env_file(dotenv_path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
                if not m:
                    continue
                key = m.group(1)
                val = m.group(2).strip()
                if (val.startswith("\"") and val.endswith("\"")) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                env[key] = val
    except FileNotFoundError:
        pass
    except Exception as e:
        env["__ERROR__"] = f"Failed to read .env: {e}"
    return env


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    suggestion: Optional[str] = None

    def to_markdown(self) -> str:
        icon = "✅" if self.passed else "❌"
        kv_parts = []
        for k, v in self.details.items():
            if isinstance(v, float):
                kv_parts.append(f"{k}={v:.4f}")
            else:
                kv_parts.append(f"{k}={v}")
        kv = ", ".join(kv_parts) if kv_parts else ""
        msg = self.message
        if kv:
            msg = f"{msg} ({kv})"
        if (not self.passed) and self.suggestion:
            msg += f"\n   - 建议: {self.suggestion}"
        return f"- {icon} {self.name}: {msg}"


def _safe_len(obj: Any) -> int:
    try:
        return int(len(obj))
    except Exception:
        return 0


# ---------------------- checks ----------------------


def check_env_backend(project_root: Path) -> CheckResult:
    dotenv = project_root / ".env"
    env_from_file = read_env_file(str(dotenv))
    backend = os.getenv("LLM_BACKEND") or env_from_file.get("LLM_BACKEND", "<unset>")
    model = os.getenv("MODEL_NAME") or env_from_file.get("MODEL_NAME", "<unset>")

    # keys (do not expose value)
    key_names = [
        "OPENAI_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "TONGYI_API_KEY",
    ]
    detected_keys = {}
    for k in key_names:
        detected_keys[k] = bool(os.getenv(k) or env_from_file.get(k))

    # dashscope import if backend == tongyi
    can_import_dashscope = None
    if str(backend).lower() == "tongyi":
        try:
            __import__("dashscope")
            can_import_dashscope = True
        except Exception:
            can_import_dashscope = False

    # llm.py contains keyword "tongyi"
    llm_path = project_root / "recjudge" / "llm.py"
    contains_tongyi = False
    try:
        text = llm_path.read_text(encoding="utf-8")
        contains_tongyi = ("tongyi" in text.lower())
    except Exception:
        contains_tongyi = False

    passed = True
    message = "backend/model/keys detected"
    if str(backend).lower() == "tongyi" and not bool(can_import_dashscope):
        passed = False
        message = "backend=tongyi but cannot import dashscope"
    if not contains_tongyi:
        # Not strictly failing, but per requirement we flag as failed if support missing
        passed = False
        message = (message + "; llm.py lacks 'tongyi' backend keyword").strip()

    details = {
        "backend": backend,
        "model": model,
        **{f"has_{k}": v for k, v in detected_keys.items()},
    }
    if can_import_dashscope is not None:
        details["import_dashscope"] = bool(can_import_dashscope)

    suggestion = None
    if not contains_tongyi:
        suggestion = "更新 recjudge/llm.py 以支持 'tongyi' 或将 LLM_BACKEND 设置为 'qwen' (DashScope 兼容)。"
    elif str(backend).lower() == "tongyi" and not bool(can_import_dashscope):
        suggestion = "pip install dashscope -U"

    return CheckResult(
        name="环境变量与后端",
        passed=passed,
        message=message,
        details=details,
        suggestion=suggestion,
    )


def _stat_parquet(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df))}
    if name == "interactions":
        out["n_users"] = int(df["user_id"].nunique()) if "user_id" in df.columns else None
        out["n_items"] = int(df["item_id"].nunique()) if "item_id" in df.columns else None
    elif name == "items":
        out["n_pop_bin"] = int(df["pop_bin"].nunique()) if "pop_bin" in df.columns else None
        out["n_age_bin"] = int(df["age_bin"].nunique()) if "age_bin" in df.columns else None
    elif name == "propensity":
        out["n_users"] = int(df["user_id"].nunique()) if "user_id" in df.columns else None
        out["n_items"] = int(df["item_id"].nunique()) if "item_id" in df.columns else None
    elif name == "judge_clean":
        if "valid_ratio" in df.columns:
            out["valid_ratio_mean"] = float(df["valid_ratio"].mean())
    elif name == "synth_tail":
        if "pop_bin" in df.columns:
            tail_ratio = float((df["pop_bin"].astype(int) >= 3).mean())
            out["tail_ratio_ge3"] = tail_ratio
    return out


def check_files(project_root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    data_proc = project_root / "data" / "proc"
    data_props = project_root / "data" / "props"
    data_judge = project_root / "data" / "judge"
    reports_dir = project_root / "reports"

    paths = {
        "interactions": data_proc / "ml1m_interactions.parquet",
        "items": data_proc / "ml1m_items.parquet",
        "propensity": data_props / "propensity.parquet",
        "pairs": data_judge / "pairs.parquet",
        "judge_clean": data_judge / "judge_clean.parquet",
        "rjs_csv": reports_dir / "tables" / "rjs_ml1m.csv",
        "corr_csv": reports_dir / "tables" / "corr_ml1m.csv",
        "calib_png": reports_dir / "figs" / "calib_isotonic.png",
        "synth_tail": project_root / "data" / "synth" / "ml1m_tail_aug.parquet",
    }

    # interactions
    for key, p in paths.items():
        try:
            if not p.exists():
                # Optional hints
                suggestion = None
                if key == "propensity":
                    suggestion = "python calib/run_ips.py --config configs/movielens1m.yaml"
                elif key == "pairs":
                    suggestion = "python recjudge/pair_sampler.py --config configs/movielens1m.yaml"
                elif key == "judge_clean":
                    suggestion = (
                        "先运行 recjudge/run_judge.py，或确认输出路径；本脚本读取 data/judge/judge_clean.parquet"
                    )
                elif key == "rjs_csv":
                    suggestion = "python recjudge/rjs_metric.py"
                results.append(
                    CheckResult(
                        name=f"文件: {key}",
                        passed=False,
                        message=f"缺失: {p}",
                        details={},
                        suggestion=suggestion,
                    )
                )
                continue

            # If exists and is parquet to be summarized
            details: Dict[str, Any] = {"path": str(p)}
            if key in {"interactions", "items", "propensity", "pairs", "judge_clean", "synth_tail"}:
                try:
                    df = pd.read_parquet(p)
                    details.update(_stat_parquet(df, key))
                    results.append(CheckResult(name=f"文件: {key}", passed=True, message="存在", details=details))
                except Exception as e:
                    results.append(
                        CheckResult(
                            name=f"文件: {key}",
                            passed=False,
                            message=f"读取失败: {e}",
                            details={"path": str(p)},
                            suggestion=None,
                        )
                    )
            else:
                results.append(CheckResult(name=f"文件: {key}", passed=True, message="存在", details=details))
        except Exception as e:
            results.append(
                CheckResult(
                    name=f"文件: {key}", passed=False, message=f"检查异常: {e}", details={"path": str(p)}
                )
            )

    return results


def check_connectivity_metrics(project_root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    judge_path = project_root / "data" / "judge" / "judge_clean.parquet"
    props_path = project_root / "data" / "props" / "propensity.parquet"

    # mini-RJS
    try:
        if not judge_path.exists():
            raise FileNotFoundError(str(judge_path))
        judge_df = pd.read_parquet(judge_path)
        if len(judge_df) == 0:
            raise ValueError("judge_clean.parquet 为空")

        try:
            from recjudge.rjs_metric import rjs_tau, pair_auc_ips  # type: ignore

            sample = judge_df.sample(n=min(100, len(judge_df)), random_state=42)
            mini = float(rjs_tau(sample, pd.read_parquet(props_path)) if props_path.exists() else np.nan)
            results.append(
                CheckResult(
                    name="连通性: mini-RJS",
                    passed=bool(np.isfinite(mini)),
                    message="完成" if np.isfinite(mini) else "跳过(无propensity)",
                    details={"mini_rjs_tau": mini},
                )
            )

            # simplified PairAUC_IPS on up to 5k pairs if props exists
            if props_path.exists():
                sample2 = judge_df.sample(n=min(5000, len(judge_df)), random_state=123)
                pair_auc = float(pair_auc_ips(sample2, pd.read_parquet(props_path)))
                results.append(
                    CheckResult(
                        name="连通性: PairAUC_IPS(5k)",
                        passed=bool(np.isfinite(pair_auc)),
                        message="完成",
                        details={"pair_auc_ips": pair_auc},
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="连通性: PairAUC_IPS(5k)",
                        passed=False,
                        message="缺少 propensity，跳过",
                        details={},
                        suggestion="python calib/run_ips.py --config configs/movielens1m.yaml",
                    )
                )
        except Exception as e:
            results.append(
                CheckResult(
                    name="连通性: 指标函数导入",
                    passed=False,
                    message=f"导入 recjudge.rjs_metric 失败: {e}",
                    details={},
                )
            )
    except Exception as e:
        results.append(
            CheckResult(
                name="连通性: mini-RJS",
                passed=False,
                message=f"读取 judge_clean 失败: {e}",
                details={},
            )
        )
    return results


def check_git_status(project_root: Path) -> CheckResult:
    try:
        def run(cmd: List[str]) -> Tuple[int, str]:
            try:
                out = subprocess.check_output(cmd, cwd=str(project_root), stderr=subprocess.STDOUT)
                return 0, out.decode("utf-8", errors="ignore").strip()
            except subprocess.CalledProcessError as e:
                return e.returncode, e.output.decode("utf-8", errors="ignore").strip()
            except Exception as e:
                return 1, str(e)

        code_branch, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        code_log, last_msg = run(["git", "log", "-1", "--pretty=%B"])
        code_stat, porcelain = run(["git", "status", "--porcelain"])

        dirty = bool(porcelain)
        passed = (code_branch == 0 and code_log == 0)
        details = {"branch": branch, "last_commit": last_msg.splitlines()[0] if last_msg else "", "dirty": dirty}
        msg = "ok" if passed else "git 信息不可用"
        return CheckResult(name="Git 状态", passed=passed, message=msg, details=details)
    except Exception as e:
        return CheckResult(name="Git 状态", passed=False, message=str(e), details={})


# ---------------------- rendering & IO ----------------------


def render_console(checks: List[CheckResult], pretty: bool) -> str:
    lines: List[str] = []
    if pretty:
        lines.append("### 进度仪表盘")
    for c in checks:
        lines.append(c.to_markdown())
    text = "\n".join(lines)
    print(text)
    return text


def save_reports(project_root: Path, checks: List[CheckResult], console_text: str) -> None:
    ensure_dir(str(project_root / "reports"))
    ensure_dir(str(project_root / "reports" / "tables"))
    ensure_dir(str(project_root / "reports" / "figs"))

    md_path = project_root / "reports" / "progress.md"
    json_path = project_root / "reports" / "progress.json"

    # Markdown: reuse console text
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(console_text + "\n")
    except Exception:
        pass

    # JSON: structured
    try:
        payload = {
            "summary": {
                "passed": int(sum(1 for c in checks if c.passed)),
                "total": int(len(checks)),
            },
            "checks": [asdict(c) for c in checks],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------- main ----------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Project health check dashboard")
    parser.add_argument("--pretty", action="store_true", help="Print with markdown heading")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    checks: List[CheckResult] = []

    # 1) Env + backend
    try:
        checks.append(check_env_backend(project_root))
    except Exception as e:
        checks.append(CheckResult(name="环境变量与后端", passed=False, message=str(e), details={}))

    # 2) Files
    try:
        checks.extend(check_files(project_root))
    except Exception as e:
        checks.append(CheckResult(name="关键文件存在性与规模", passed=False, message=str(e), details={}))

    # 3) Connectivity + light metrics
    try:
        checks.extend(check_connectivity_metrics(project_root))
    except Exception as e:
        checks.append(CheckResult(name="连通性与轻量指标", passed=False, message=str(e), details={}))

    # 4) Git status (optional)
    try:
        checks.append(check_git_status(project_root))
    except Exception:
        pass

    # Render & save
    console_text = render_console(checks, pretty=args.pretty)
    save_reports(project_root, checks, console_text)


if __name__ == "__main__":
    main()


