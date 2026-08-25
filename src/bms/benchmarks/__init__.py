"""Benchmark suite: published methods run through this project's own gate.

    from src.bms.benchmarks import load_all, run_study

    methods = load_all()                  # registry, populated
    results = run_study(frame, target="cumulative_fade")

See `registry.py` for why availability is reported rather than skipped, and
`targets.py` for why the target definition is treated as a variable.
"""

from src.bms.benchmarks.registry import (
    Availability,
    BenchmarkMethod,
    Family,
    all_methods,
    check_availability,
    get,
    load_all,
    register,
    registry_frame,
)
from src.bms.benchmarks.study import (
    MethodResult,
    StudyResult,
    run_study,
)
from src.bms.benchmarks.targets import (
    CellScreen,
    SignalEstimate,
    add_targets,
    screen_cells_for_soh,
    signal_report,
    signal_to_noise,
    target_columns,
)

__all__ = [
    "Availability",
    "BenchmarkMethod",
    "CellScreen",
    "Family",
    "MethodResult",
    "SignalEstimate",
    "StudyResult",
    "add_targets",
    "all_methods",
    "check_availability",
    "get",
    "load_all",
    "register",
    "registry_frame",
    "run_study",
    "screen_cells_for_soh",
    "signal_report",
    "signal_to_noise",
    "target_columns",
]
