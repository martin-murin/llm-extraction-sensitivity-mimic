"""
Provides shared helpers/configuration for publication figure modules.

Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
Writes: local reports/artifacts determined by CLI defaults.
Supports publication figure generation.
"""

__all__ = [
    "config",
    "data_loaders",
    "plot_utils",
]
