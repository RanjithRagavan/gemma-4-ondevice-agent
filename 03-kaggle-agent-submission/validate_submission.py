#!/usr/bin/env python3
"""
Strict offline validator mirroring swegemma / adk-submission validation rules
(HARNESS_README.md Sections 2, 3, and 7).

Validates a submission directory OR submission.zip archive for:
1. Root config discovery (exactly one agent.yaml at archive root)
2. Strictly allowed file extensions & zero symlinks / path traversal ('..')
3. Recursive !include resolution & cycle detection
4. Single Base Model rule against ALLOWED_MODEL_NAMES
5. Built-in tool names & AgentTool config references
6. GenerationConstraints & vLLM thinking_config best practices
7. ADK session.state curly-brace templating safety (prevents KeyError crashes!)
8. ADK Skill SKILL.md YAML frontmatter & Python script AST compilation
"""

from __future__ import annotations

import ast
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}

ALLOWED_TOOLS = {
    "run_command",
    "submit_patch",
    "get_status",
    "read_file",
    "edit_file",
    "write_file",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
}

ALLOWED_EXTENSIONS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
ALLOWED_GEN_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop_sequences",
    "response_mime_type",
    "seed",
    "thinking_config",
}
ALLOWED_ADK_STATE_VARS = {"problem_description", "hints", "hints?"}
MAX_TOTAL_BYTES = 3_221_225_472  # 3 GiB


class IncludeLoader(yaml.SafeLoader):
    def __init__(self, stream: Any, base_dir: Path, root_dir: Path, depth: int = 0) -> None:
        self._base_dir = base_dir
        self._root_dir = root_dir
        self._depth = depth
        super().__init__(stream)


def _construct_include(loader: IncludeLoader, node: yaml.Node) -> Any:
    rel_str = loader.construct_scalar(node)
    if "\x00" in rel_str or rel_str.startswith("/") or ".." in Path(rel_str).parts:
        raise ValueError(f"PathTraversalError: forbidden !include path '{rel_str}'")
    if loader._depth >= 10:
        raise ValueError(f"Max !include depth (10) exceeded at '{rel_str}'")

    target = (loader._base_dir / rel_str).resolve()
    if not target.is_relative_to(loader._root_dir.resolve()):
        raise ValueError(f"PathTraversalError: '{rel_str}' escapes submission root")
    if not target.is_file():
        raise FileNotFoundError(f"!include target not found: {target}")

    if target.suffix in (".md", ".txt"):
        return target.read_text(encoding="utf-8")
    elif target.suffix in (".yaml", ".yml"):
        with target.open("r", encoding="utf-8") as fh:
            sub_loader = IncludeLoader(fh, target.parent, loader._root_dir, loader._depth + 1)
            try:
                return sub_loader.get_single_data()
            finally:
                sub_loader.dispose()
    else:
        raise ValueError(f"Unsupported !include file extension: {target.suffix}")


IncludeLoader.add_constructor("!include", _construct_include)


def load_yaml_with_includes(yaml_path: Path, root_dir: Path) -> Dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as fh:
        loader = IncludeLoader(fh, yaml_path.parent, root_dir, depth=0)
        try:
            data = loader.get_single_data()
        finally:
            loader.dispose()
    if not isinstance(data, dict):
        raise ValueError(f"{yaml_path} did not parse into a YAML mapping")
    return data


def validate_agent_tree(
    yaml_path: Path,
    root_dir: Path,
    models_seen: Set[str],
    errors: List[str],
    warnings: List[str],
    visited: Set[Path],
) -> None:
    resolved = yaml_path.resolve()
    if resolved in visited:
        return
    visited.add(resolved)

    try:
        cfg = load_yaml_with_includes(yaml_path, root_dir)
    except Exception as exc:
        errors.append(f"Failed loading {yaml_path.relative_to(root_dir)}: {exc}")
        return

    agent_class = cfg.get("agent_class", "LlmAgent")
    if agent_class not in {"LlmAgent", "SequentialAgent", "ParallelAgent", "LoopAgent"}:
        errors.append(f"{yaml_path}: unsupported agent_class '{agent_class}'")

    if agent_class == "LlmAgent":
        model = cfg.get("model")
        if not model:
            errors.append(f"{yaml_path}: missing required 'model'")
        else:
            models_seen.add(model)
            if model not in ALLOWED_MODELS:
                errors.append(f"{yaml_path}: model '{model}' not in ALLOWED_MODELS {sorted(ALLOWED_MODELS)}")

        # Validate ADK instruction state placeholders
        instr = cfg.get("instruction", "")
        if isinstance(instr, str):
            for match in re.finditer(r"\{([a-zA-Z_][a-zA-Z0-9_]*\??)\}", instr):
                var = match.group(1)
                if var not in ALLOWED_ADK_STATE_VARS:
                    errors.append(
                        f"{yaml_path.relative_to(root_dir)}: ADK instruction contains bare placeholder '{{{var}}}' "
                        "which will raise KeyError unless present in session.state!"
                    )

        # Validate generate_content_config
        gen = cfg.get("generate_content_config") or {}
        if isinstance(gen, dict):
            for k in gen:
                if k not in ALLOWED_GEN_FIELDS:
                    errors.append(f"{yaml_path}: forbidden generate_content_config field '{k}'")
            if gen.get("max_output_tokens", 1) > 32768:
                errors.append(f"{yaml_path}: max_output_tokens exceeds 32768")
            tc = gen.get("thinking_config") or {}
            if isinstance(tc, dict):
                if "thinking_level" in tc:
                    warnings.append(f"{yaml_path}: omit thinking_level for vLLM compatibility")
                if tc.get("thinking_budget", 0) > 32768:
                    errors.append(f"{yaml_path}: thinking_budget exceeds 32768")

        # Validate skills
        for sk in cfg.get("skills") or []:
            if ".." in Path(sk).parts or sk.startswith("/"):
                errors.append(f"{yaml_path}: invalid skill path '{sk}'")
                continue
            sk_dir = root_dir / sk
            sk_md = sk_dir / "SKILL.md"
            if not sk_md.is_file():
                errors.append(f"{yaml_path}: skill '{sk}' missing SKILL.md")
            else:
                text = sk_md.read_text(encoding="utf-8")
                m = re.match(r"^---\s*\n(.*?)\n---", text, flags=re.S)
                if not m:
                    errors.append(f"{sk_md}: missing YAML frontmatter")
                else:
                    fm = yaml.safe_load(m.group(1))
                    if not isinstance(fm, dict) or not fm.get("name"):
                        errors.append(f"{sk_md}: frontmatter missing 'name'")

        # Validate tools
        for tool in cfg.get("tools") or []:
            if isinstance(tool, str):
                if tool not in ALLOWED_TOOLS:
                    errors.append(f"{yaml_path}: unknown tool '{tool}'")
            elif isinstance(tool, dict) and "agent_tool" in tool:
                at = tool["agent_tool"]
                sub_rel = at.get("config_path", "")
                if not sub_rel or ".." in Path(sub_rel).parts or sub_rel.startswith("/"):
                    errors.append(f"{yaml_path}: invalid agent_tool config_path '{sub_rel}'")
                else:
                    sub_path = root_dir / sub_rel
                    if not sub_path.is_file():
                        errors.append(f"{yaml_path}: missing agent_tool config '{sub_rel}'")
                    else:
                        validate_agent_tree(sub_path, root_dir, models_seen, errors, warnings, visited)
            else:
                errors.append(f"{yaml_path}: unrecognized tool entry {tool!r}")

    for sub in cfg.get("sub_agents") or []:
        sub_rel = sub.get("config_path", "") if isinstance(sub, dict) else ""
        if sub_rel:
            validate_agent_tree(root_dir / sub_rel, root_dir, models_seen, errors, warnings, visited)


def validate_submission_dir(root_dir: Path) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Root config check
    root_candidates = [
        root_dir / name for name in ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
        if (root_dir / name).is_file()
    ]
    if len(root_candidates) != 1:
        errors.append(f"Expected exactly 1 root config at archive root, found {len(root_candidates)}: {root_candidates}")
        return False, errors, warnings

    # 2. File extension, symlink, and size checks
    total_bytes = 0
    file_count = 0
    for p in root_dir.rglob("*"):
        if p.is_symlink():
            errors.append(f"Symlink forbidden: {p.relative_to(root_dir)}")
        if p.is_file():
            file_count += 1
            total_bytes += p.stat().st_size
            if p.suffix not in ALLOWED_EXTENSIONS:
                errors.append(f"Disallowed file extension '{p.suffix}' on {p.relative_to(root_dir)}")
            if p.suffix == ".py":
                try:
                    ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
                except SyntaxError as exc:
                    errors.append(f"SyntaxError in {p.relative_to(root_dir)}:{exc.lineno}: {exc.msg}")

    if total_bytes >= MAX_TOTAL_BYTES:
        errors.append(f"Total unpacked size {total_bytes} exceeds 3 GiB limit")

    # 3. Optional eval_config.yaml check
    eval_cfg_path = root_dir / "eval_config.yaml"
    if eval_cfg_path.is_file():
        ec = yaml.safe_load(eval_cfg_path.read_text(encoding="utf-8"))
        if not isinstance(ec, dict) or "evaluation" not in ec:
            errors.append("eval_config.yaml must contain top-level 'evaluation:' mapping")

    # 4. Validate agent tree & single-model rule
    models_seen: Set[str] = set()
    validate_agent_tree(root_candidates[0], root_dir, models_seen, errors, warnings, set())
    if len(models_seen) != 1:
        errors.append(f"Single Base Model Rule violated! Declared models: {sorted(models_seen)}")

    return len(errors) == 0, errors, warnings


def validate_path(target: Path) -> bool:
    if target.is_file() and target.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(target, "r") as zf:
                for info in zf.infolist():
                    if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                        print(f"[FAIL] Zip slip / bad path in archive: {info.filename}")
                        return False
                zf.extractall(tmp)
            ok, errs, warns = validate_submission_dir(Path(tmp))
    else:
        ok, errs, warns = validate_submission_dir(target)

    for w in warns:
        print(f"  [WARN] {w}")
    for e in errs:
        print(f"  [ERROR] {e}")
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Validation for {target}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_submission.py <submission_dir_or_zip> [...]")
        sys.exit(1)
    all_ok = all(validate_path(Path(arg)) for arg in sys.argv[1:])
    sys.exit(0 if all_ok else 1)
