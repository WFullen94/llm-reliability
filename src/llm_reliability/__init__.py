"""llm-reliability: calibration, drift detection, and adversarial robustness for LLMs."""

__version__ = "0.1.0"

from llm_reliability.calibration import (
    CalibrationResult,
    BinStats,
    calibration_result,
    ece,
    mce,
    reliability_diagram,
    verbalized,
    temperature_scale,
    apply_temperature,
    conformal_threshold,
)

__all__ = [
    "CalibrationResult",
    "BinStats",
    "calibration_result",
    "ece",
    "mce",
    "reliability_diagram",
    "verbalized",
    "temperature_scale",
    "apply_temperature",
    "conformal_threshold",
]
