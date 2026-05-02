"""llm-reliability: calibration, drift detection, and adversarial robustness for LLMs."""

__version__ = "0.3.0"

from llm_reliability.drift import (
    DriftSnapshot,
    DriftTest,
    DriftResult,
    ChangedExample,
    capture,
    compare,
)

from llm_reliability.adapters import (
    ModelResponse,
    ModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    OllamaAdapter,
    MLXAdapter,
    HuggingFaceAdapter,
)

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
    semantic_entropy,
    semantic_dispersion,
    consistency,
    p_true,
)

__all__ = [
    "DriftSnapshot",
    "DriftTest",
    "DriftResult",
    "ChangedExample",
    "capture",
    "compare",
    "ModelResponse",
    "ModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
    "MLXAdapter",
    "HuggingFaceAdapter",
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
    "semantic_entropy",
    "semantic_dispersion",
    "consistency",
    "p_true",
]
