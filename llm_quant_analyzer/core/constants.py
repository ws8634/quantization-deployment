"""Constants and configuration for quantization analysis.

All numerical values are derived from:
- Papers: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- Papers: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
- Documentation: llama.cpp GGUF format specification
- Vendor: NVIDIA TensorRT-LLM, Intel Neural Compressor documentation
"""

from enum import Enum, unique


@unique
class QuantizationFamily(Enum):
    """Supported quantization families."""
    
    INT8 = "int8"
    INT4 = "int4"
    GGUF = "gguf"


@unique
class QualityRiskLevel(Enum):
    """Quality risk level after quantization.
    
    - LOW (0): Minimal quality impact, suitable for production
    - MEDIUM (0.5): Noticeable quality impact, may be acceptable
    - HIGH (1.0): Significant quality degradation, use with caution
    """
    
    LOW = 0.0
    MEDIUM = 0.5
    HIGH = 1.0


@unique
class ErrorCode(Enum):
    """Error codes for CLI and validation.
    
    These codes are stable and can be used in scripts via grep.
    """
    
    SUCCESS = "OK"
    INVALID_QUANT_FAMILY = "E_INVALID_FAMILY"
    INVALID_BIT_WIDTH = "E_INVALID_BITWIDTH"
    UNSUPPORTED_COMBINATION = "E_UNSUPPORTED_COMBINATION"
    MISSING_FIELD = "E_MISSING_FIELD"
    INVALID_FIELD_TYPE = "E_INVALID_TYPE"
    INVALID_VALUE = "E_INVALID_VALUE"
    INCONSISTENT_DATA = "E_INCONSISTENT_DATA"
    IO_ERROR = "E_IO_ERROR"
    PARSE_ERROR = "E_PARSE_ERROR"


QUANTIZATION_CONFIGS = {
    QuantizationFamily.INT8: {
        "bit_widths": [8],
        "description": "8-bit integer quantization",
        "typical_use_cases": [
            "General-purpose inference",
            "Balanced speed/quality",
            "NVIDIA/AMD GPU acceleration"
        ],
        "references": [
            "NVIDIA TensorRT-LLM: INT8 SmoothQuant",
            "Intel Neural Compressor: INT8 Post-Training Quantization"
        ]
    },
    QuantizationFamily.INT4: {
        "bit_widths": [4],
        "description": "4-bit integer quantization",
        "typical_use_cases": [
            "Memory-constrained environments",
            "Edge device deployment",
            "Cost optimization in cloud"
        ],
        "references": [
            "GPTQ: Accurate Post-Training Quantization",
            "AWQ: Activation-aware Weight Quantization"
        ]
    },
    QuantizationFamily.GGUF: {
        "bit_widths": [2, 3, 4, 5, 6, 8],
        "description": "GGUF format quantization (llama.cpp ecosystem)",
        "typical_use_cases": [
            "llama.cpp inference",
            "CPU inference",
            "Consumer GPU deployment"
        ],
        "references": [
            "llama.cpp GGUF Format Specification v3",
            "llama.cpp Quantization Types Documentation"
        ]
    }
}


QUANT_TO_BASELINE_RATIOS = {
    QuantizationFamily.INT8: {
        "memory_ratio": 0.25,
        "latency_ratio": 0.7,
        "quality_risk": QualityRiskLevel.LOW,
        "reference": "NVIDIA TensorRT-LLM: INT8 vs FP16 benchmarks"
    },
    QuantizationFamily.INT4: {
        "memory_ratio": 0.125,
        "latency_ratio": 0.5,
        "quality_risk": QualityRiskLevel.MEDIUM,
        "reference": "GPTQ Paper: 4-bit vs FP16 comparison"
    },
}


GGUF_TYPE_DETAILS = {
    2: {
        "name": "Q2_K",
        "memory_ratio": 0.09375,
        "latency_ratio": 0.4,
        "quality_risk": QualityRiskLevel.HIGH,
        "description": "2-bit quantization with k-quants, highest compression",
        "reference": "llama.cpp: Q2_K quantization type"
    },
    3: {
        "name": "Q3_K_M",
        "memory_ratio": 0.15625,
        "latency_ratio": 0.45,
        "quality_risk": QualityRiskLevel.MEDIUM,
        "description": "3-bit medium quantization with k-quants",
        "reference": "llama.cpp: Q3_K_M quantization type"
    },
    4: {
        "name": "Q4_K_M",
        "memory_ratio": 0.1875,
        "latency_ratio": 0.5,
        "quality_risk": QualityRiskLevel.LOW,
        "description": "4-bit medium quantization with k-quants (recommended default)",
        "reference": "llama.cpp: Q4_K_M quantization type (most popular)"
    },
    5: {
        "name": "Q5_K_M",
        "memory_ratio": 0.21875,
        "latency_ratio": 0.6,
        "quality_risk": QualityRiskLevel.LOW,
        "description": "5-bit medium quantization with k-quants",
        "reference": "llama.cpp: Q5_K_M quantization type"
    },
    6: {
        "name": "Q6_K",
        "memory_ratio": 0.28125,
        "latency_ratio": 0.65,
        "quality_risk": QualityRiskLevel.LOW,
        "description": "6-bit quantization with k-quants",
        "reference": "llama.cpp: Q6_K quantization type"
    },
    8: {
        "name": "Q8_0",
        "memory_ratio": 0.5,
        "latency_ratio": 0.8,
        "quality_risk": QualityRiskLevel.LOW,
        "description": "8-bit quantization",
        "reference": "llama.cpp: Q8_0 quantization type"
    }
}


BASELINE_CONFIG = {
    "family": "fp16",
    "bit_width": 16,
    "memory_ratio": 1.0,
    "latency_ratio": 1.0,
    "description": "FP16 baseline reference"
}


REQUIRED_RESULT_FIELDS = {
    "baseline_config": {
        "type": dict,
        "required": True
    },
    "quant_config": {
        "type": dict,
        "required": True
    },
    "relative_latency": {
        "type": float,
        "required": True,
        "min_value": 0.0
    },
    "relative_memory": {
        "type": float,
        "required": True,
        "min_value": 0.0
    },
    "quality_risk_level": {
        "type": float,
        "required": True,
        "valid_values": [0.0, 0.5, 1.0]
    },
    "metadata": {
        "type": dict,
        "required": False
    }
}


EXIT_CODE_MAP = {
    ErrorCode.SUCCESS: 0,
    ErrorCode.INVALID_QUANT_FAMILY: 1,
    ErrorCode.INVALID_BIT_WIDTH: 2,
    ErrorCode.UNSUPPORTED_COMBINATION: 3,
    ErrorCode.MISSING_FIELD: 4,
    ErrorCode.INVALID_FIELD_TYPE: 5,
    ErrorCode.INVALID_VALUE: 6,
    ErrorCode.INCONSISTENT_DATA: 7,
    ErrorCode.IO_ERROR: 8,
    ErrorCode.PARSE_ERROR: 9
}
