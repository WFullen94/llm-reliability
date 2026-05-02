"""llm-reliability: calibration, drift detection, and adversarial robustness for LLMs."""

__version__ = "0.2.0"

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
