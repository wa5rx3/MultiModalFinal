from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

# Optional deps used if available
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".tsv", ".sql", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".r", ".ipynb"
}

CODE_EXTENSIONS = {
    ".py", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".sql", ".r"
}

SKIP_DIRS_DEFAULT = {
    ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".idea", ".vscode", "node_modules"
}

BINARY_EXTENSIONS = {
    ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".pickle", ".png", ".jpg", ".jpeg",
    ".webp", ".gif", ".pdf", ".zip", ".gz", ".7z", ".rar", ".exe", ".dll", ".so"
}


@dataclass
class DigestConfig:
    root: Path
    output: Path
    max_text_chars: int
    max_csv_rows: int
    max_parquet_rows: int
    max_tree_entries: int
    include_hashes: bool
    skip_dirs: set[str]
    include_hidden: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as e:
            return f"[UNREADABLE TEXT FILE: {e}]"
    except Exception as e:
        return f"[UNREADABLE TEXT FILE: {e}]"

    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[TRUNCATED: original length={len(text)} chars]"
    return text


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    return f"{size:.2f} {units[unit]}"


def modified_time(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return "unknown"


def file_header(path: Path, cfg: DigestConfig) -> str:
    stat = path.stat()
    bits = [
        f"Path: {path.relative_to(cfg.root)}",
        f"Size: {format_size(stat.st_size)}",
        f"Modified: {modified_time(path)}",
    ]
    if cfg.include_hashes and stat.st_size <= 1024 * 1024 * 1024:
        try:
            bits.append(f"SHA256: {sha256_file(path)}")
        except Exception as e:
            bits.append(f"SHA256: [ERROR: {e}]")
    return "\n".join(bits)


def iter_files(root: Path, cfg: DigestConfig) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)

        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in cfg.skip_dirs and (cfg.include_hidden or not d.startswith("."))
        ]

        if not cfg.include_hidden and is_hidden(dir_path.relative_to(root)):
            continue

        for fname in sorted(filenames):
            p = dir_path / fname
            if not cfg.include_hidden and is_hidden(p.relative_to(root)):
                continue
            files.append(p)
    return files


def build_tree(root: Path, cfg: DigestConfig) -> str:
    lines = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        rel_dir = dir_path.relative_to(root)

        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in cfg.skip_dirs and (cfg.include_hidden or not d.startswith("."))
        ]
        filenames = [
            f for f in sorted(filenames)
            if cfg.include_hidden or not f.startswith(".")
        ]

        depth = 0 if str(rel_dir) == "." else len(rel_dir.parts)
        indent = "  " * depth
        lines.append(f"{indent}{dir_path.name}/")
        count += 1
        if count >= cfg.max_tree_entries:
            lines.append("[TREE TRUNCATED]")
            break

        for fname in filenames:
            lines.append(f"{indent}  {fname}")
            count += 1
            if count >= cfg.max_tree_entries:
                lines.append("[TREE TRUNCATED]")
                return "\n".join(lines)

    return "\n".join(lines)


def summarize_json(path: Path) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        return pretty
    except Exception as e:
        return f"[JSON PARSE ERROR: {e}]"


def summarize_csv(path: Path, cfg: DigestConfig) -> str:
    lines = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i + 1 >= cfg.max_csv_rows:
                    break
        lines.append(f"Preview rows shown: {len(rows)}")
        for i, row in enumerate(rows):
            lines.append(f"ROW {i}: {row}")
        if pd is not None:
            try:
                df = pd.read_csv(path, nrows=cfg.max_csv_rows)
                lines.append(f"Columns: {list(df.columns)}")
            except Exception:
                pass
    except Exception as e:
        return f"[CSV READ ERROR: {e}]"
    return "\n".join(lines)


def summarize_ipynb(path: Path, max_chars: int) -> str:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"[NOTEBOOK READ ERROR: {e}]"

    cells = nb.get("cells", [])
    out = [f"Notebook cells: {len(cells)}"]
    total_chars = 0

    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))
        block = f"\n--- CELL {i} ({ctype}) ---\n{source}"
        if total_chars + len(block) > max_chars:
            out.append("\n[NOTEBOOK CONTENT TRUNCATED]")
            break
        out.append(block)
        total_chars += len(block)

    return "\n".join(out)


def summarize_parquet(path: Path, cfg: DigestConfig) -> str:
    parts = []

    if pq is not None:
        try:
            pf = pq.ParquetFile(path)
            meta = pf.metadata
            parts.append(f"Num rows: {meta.num_rows}")
            parts.append(f"Num row groups: {meta.num_row_groups}")
            parts.append(f"Schema:")
            schema_str = str(pf.schema)
            parts.append(schema_str)
        except Exception as e:
            parts.append(f"[PYARROW PARQUET METADATA ERROR: {e}]")

    if pd is not None:
        try:
            df = pd.read_parquet(path)
            parts.append(f"Pandas rows: {len(df)}")
            parts.append(f"Columns ({len(df.columns)}): {list(df.columns)}")
            nulls = df.isna().sum().sort_values(ascending=False)
            parts.append("Top null counts:")
            parts.append(nulls.head(20).to_string())

            parts.append("Dtypes:")
            parts.append(df.dtypes.astype(str).to_string())

            sample = df.head(cfg.max_parquet_rows)
            parts.append(f"Head({cfg.max_parquet_rows}):")
            parts.append(sample.to_string(max_rows=cfg.max_parquet_rows, max_cols=50))

            # Numeric summary if available
            try:
                num = df.select_dtypes(include=["number", "bool"])
                if not num.empty:
                    desc = num.describe(include="all").transpose().head(20)
                    parts.append("Numeric summary (first 20 numeric/bool columns):")
                    parts.append(desc.to_string())
            except Exception:
                pass
        except Exception as e:
            parts.append(f"[PANDAS PARQUET READ ERROR: {e}]")

    return "\n".join(parts)


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".ipynb":
        return "ipynb"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in BINARY_EXTENSIONS:
        return "binary"
    return "other"


def section(title: str, body: str) -> str:
    line = "=" * 100
    return f"\n{line}\n{title}\n{line}\n{body}\n"


def repo_summary(files: list[Path], cfg: DigestConfig) -> str:
    by_ext = defaultdict(int)
    total_size = 0
    for f in files:
        by_ext[f.suffix.lower() or "[no_ext]"] += 1
        try:
            total_size += f.stat().st_size
        except Exception:
            pass

    lines = [
        f"Root: {cfg.root}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Total files scanned: {len(files)}",
        f"Total size scanned: {format_size(total_size)}",
        "",
        "Counts by extension:",
    ]
    for ext, count in sorted(by_ext.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {ext}: {count}")
    return "\n".join(lines)


def important_paths_summary(files: list[Path], cfg: DigestConfig) -> str:
    interesting = []
    keywords = [
        "current_state", "runbook", "data_versions", "project_definition", "cohort_definition",
        "triage_feature_policy", "registry", "summary", "metrics", "config", "report"
    ]
    for f in files:
        rel = str(f.relative_to(cfg.root)).lower()
        if any(k in rel for k in keywords):
            interesting.append(rel)
    interesting = sorted(set(interesting))
    return "\n".join(interesting[:500]) if interesting else "[None found]"


def generate_digest(cfg: DigestConfig) -> str:
    files = iter_files(cfg.root, cfg)

    out = []
    out.append(section("REPOSITORY SUMMARY", repo_summary(files, cfg)))
    out.append(section("REPOSITORY TREE", build_tree(cfg.root, cfg)))
    out.append(section("IMPORTANT PATHS", important_paths_summary(files, cfg)))

    # Process files in deterministic order
    for path in sorted(files):
        rel = path.relative_to(cfg.root)
        kind = classify_file(path)
        header = file_header(path, cfg)

        if kind == "json":
            body = summarize_json(path)
        elif kind == "csv":
            body = summarize_csv(path, cfg)
        elif kind == "parquet":
            body = summarize_parquet(path, cfg)
        elif kind == "ipynb":
            body = summarize_ipynb(path, cfg.max_text_chars)
        elif kind == "text":
            body = safe_read_text(path, cfg.max_text_chars)
        elif kind == "binary":
            body = "[Binary file omitted from verbatim dump. See metadata above.]"
        else:
            # Try text, otherwise omit
            body = safe_read_text(path, cfg.max_text_chars)

        title = f"FILE: {rel} [{kind}]"
        out.append(section(title, f"{header}\n\n{body}"))

    return "\n".join(out)


def parse_args() -> DigestConfig:
    parser = argparse.ArgumentParser(
        description="Generate a single text digest of a project for AI review."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="project_digest.txt",
        help="Output text file",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=120000,
        help="Max chars to dump per text/code/json/notebook file",
    )
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=20,
        help="Max preview rows per CSV",
    )
    parser.add_argument(
        "--max-parquet-rows",
        type=int,
        default=10,
        help="Max head rows shown per Parquet file",
    )
    parser.add_argument(
        "--max-tree-entries",
        type=int,
        default=5000,
        help="Max entries in tree section",
    )
    parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Disable SHA256 hashes",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files/directories",
    )
    parser.add_argument(
        "--skip-dirs",
        nargs="*",
        default=[],
        help="Extra directory names to skip",
    )

    args = parser.parse_args()

    return DigestConfig(
        root=Path(args.root).resolve(),
        output=Path(args.output).resolve(),
        max_text_chars=args.max_text_chars,
        max_csv_rows=args.max_csv_rows,
        max_parquet_rows=args.max_parquet_rows,
        max_tree_entries=args.max_tree_entries,
        include_hashes=not args.no_hashes,
        skip_dirs=SKIP_DIRS_DEFAULT.union(set(args.skip_dirs)),
        include_hidden=args.include_hidden,
    )


def main() -> None:
    cfg = parse_args()

    if not cfg.root.exists():
        raise FileNotFoundError(f"Root does not exist: {cfg.root}")

    digest = generate_digest(cfg)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(digest, encoding="utf-8")

    print(f"Saved digest to: {cfg.output}")
    print(f"Root: {cfg.root}")


if __name__ == "__main__":
    main()