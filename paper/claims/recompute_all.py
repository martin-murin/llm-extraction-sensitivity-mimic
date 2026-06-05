"""Run every compute_*.py script under paper/claims/scripts/ to refresh claims.json.

Each compute script is independent and may update some subset of claim keys.
After this script completes, claims.json reflects the current data and code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_import_paths() -> None:
    """Ensure repo root and scripts dir are importable."""
    for p in (REPO_ROOT, SCRIPTS_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def main() -> int:
    _bootstrap_import_paths()

    if not SCRIPTS_DIR.is_dir():
        print(f"Scripts directory not found: {SCRIPTS_DIR}", file=sys.stderr)
        return 1

    script_paths = sorted(p for p in SCRIPTS_DIR.glob("compute_*.py"))
    if not script_paths:
        print(f"No compute_*.py scripts found in {SCRIPTS_DIR}", file=sys.stderr)
        return 1

    failures = []
    for script_path in script_paths:
        print(f"=== Running {script_path.name} ===")
        module_name = f"paper.claims.scripts.{script_path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                script_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load spec for {script_path}")
            # Ensure a clean import namespace and register the module before exec.
            # Some decorators (e.g., @dataclass) consult sys.modules during class
            # creation and fail when the module is missing from that registry.
            sys.modules.pop(module_name, None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            if hasattr(module, "main"):
                exit_code = module.main()
                if exit_code != 0:
                    failures.append((script_path.name, f"exit code {exit_code}"))
            else:
                print(f"  WARNING: {script_path.name} has no main() function; skipping")
        except Exception as exc:
            sys.modules.pop(module_name, None)
            print(f"  ERROR: {exc}", file=sys.stderr)
            failures.append((script_path.name, str(exc)))

    print()
    print(f"Ran {len(script_paths)} compute scripts.")
    if failures:
        print(f"{len(failures)} failed:")
        for name, reason in failures:
            print(f"  {name}: {reason}")
        return 1
    print("All succeeded. claims.json is current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
