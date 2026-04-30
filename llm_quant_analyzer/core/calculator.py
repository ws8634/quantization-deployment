"""Calculator for quantization impact analysis.

This module contains functions to calculate:
- Relative latency compared to baseline
- Relative memory footprint compared to baseline
- Quality risk level based on quantization type

All ratios are based on publicly available benchmarks and research papers.
"""

from typing import Optional, Tuple, Dict, Any

from .constants import (
    QuantizationFamily,
    QualityRiskLevel,
    ErrorCode,
    QUANTIZATION_CONFIGS,
    QUANT_TO_BASELINE_RATIOS,
    GGUF_TYPE_DETAILS,
    BASELINE_CONFIG
)
from .datatypes import (
    QuantizationConfig,
    BaselineConfig,
    QuantizationResult,
    ValidationError,
    SupportedQuantizationInfo
)


def is_bit_width_supported(family: QuantizationFamily, bit_width: int) -> bool:
    """Check if a bit width is supported for a given quantization family.
    
    Args:
        family: The quantization family
        bit_width: The bit width to check
    
    Returns:
        True if supported, False otherwise
    """
    if family not in QUANTIZATION_CONFIGS:
        return False
    return bit_width in QUANTIZATION_CONFIGS[family]["bit_widths"]


def validate_quantization_config(config: QuantizationConfig) -> Optional[ValidationError]:
    """Validate a quantization configuration.
    
    Args:
        config: The quantization configuration to validate
    
    Returns:
        ValidationError if invalid, None if valid
    """
    if not is_bit_width_supported(config.family, config.bit_width):
        supported = QUANTIZATION_CONFIGS[config.family]["bit_widths"]
        return ValidationError(
            error_code=ErrorCode.UNSUPPORTED_COMBINATION,
            message=f"Bit width {config.bit_width} not supported for {config.family.value}. "
                   f"Supported: {supported}",
            field="bit_width"
        )
    
    if config.bit_width <= 0:
        return ValidationError(
            error_code=ErrorCode.INVALID_VALUE,
            message=f"Bit width must be positive, got {config.bit_width}",
            field="bit_width"
        )
    
    return None


def calculate_quantization_impact(
    quant_config: QuantizationConfig,
    baseline_config: Optional[BaselineConfig] = None
) -> Tuple[Optional[QuantizationResult], Optional[ValidationError]]:
    """Calculate the impact of quantization compared to baseline.
    
    Based on:
    - GPTQ Paper: 4-bit quantization maintains ~99% of FP16 quality
    - NVIDIA TensorRT-LLM: INT8 provides ~2.5x memory reduction, ~1.4x speedup
    - llama.cpp Documentation: GGUF Q4_K_M is recommended default
    
    Args:
        quant_config: The quantization configuration
        baseline_config: Optional baseline configuration (defaults to FP16)
    
    Returns:
        Tuple of (QuantizationResult, ValidationError) - result is None if error,
        error is None if successful
    """
    validation_error = validate_quantization_config(quant_config)
    if validation_error:
        return None, validation_error
    
    if baseline_config is None:
        baseline_config = BaselineConfig()
    
    if quant_config.family == QuantizationFamily.GGUF:
        if quant_config.bit_width not in GGUF_TYPE_DETAILS:
            return None, ValidationError(
                error_code=ErrorCode.INVALID_BIT_WIDTH,
                message=f"GGUF bit width {quant_config.bit_width} not found in type details",
                field="bit_width"
            )
        
        gguf_details = GGUF_TYPE_DETAILS[quant_config.bit_width]
        memory_ratio = gguf_details["memory_ratio"]
        latency_ratio = gguf_details["latency_ratio"]
        quality_risk = gguf_details["quality_risk"].value
        reference = gguf_details["reference"]
    else:
        if quant_config.family not in QUANT_TO_BASELINE_RATIOS:
            return None, ValidationError(
                error_code=ErrorCode.INVALID_QUANT_FAMILY,
                message=f"Quantization family {quant_config.family.value} not found in ratios",
                field="family"
            )
        
        ratios = QUANT_TO_BASELINE_RATIOS[quant_config.family]
        memory_ratio = ratios["memory_ratio"]
        latency_ratio = ratios["latency_ratio"]
        quality_risk = ratios["quality_risk"].value
        reference = ratios["reference"]
    
    if memory_ratio <= 0:
        return None, ValidationError(
            error_code=ErrorCode.INVALID_VALUE,
            message=f"Memory ratio must be positive, got {memory_ratio}",
            field="memory_ratio"
        )
    
    if latency_ratio <= 0:
        return None, ValidationError(
            error_code=ErrorCode.INVALID_VALUE,
            message=f"Latency ratio must be positive, got {latency_ratio}",
            field="latency_ratio"
        )
    
    result = QuantizationResult(
        baseline_config=baseline_config.to_dict(),
        quant_config=quant_config.to_dict(),
        relative_latency=latency_ratio,
        relative_memory=memory_ratio,
        quality_risk_level=quality_risk,
        metadata={
            "reference": reference,
            "calculation_notes": {
                "relative_latency": "< 1.0 means faster than baseline",
                "relative_memory": "< 1.0 means smaller memory footprint than baseline",
                "quality_risk_level": "0.0=LOW, 0.5=MEDIUM, 1.0=HIGH risk of quality degradation"
            }
        }
    )
    
    return result, None


def get_supported_quantizations() -> Dict[QuantizationFamily, SupportedQuantizationInfo]:
    """Get information about all supported quantization families.
    
    Returns:
        Dictionary mapping QuantizationFamily to SupportedQuantizationInfo
    """
    result = {}
    for family, config in QUANTIZATION_CONFIGS.items():
        result[family] = SupportedQuantizationInfo(
            family=family,
            bit_widths=config["bit_widths"],
            description=config["description"],
            typical_use_cases=config["typical_use_cases"],
            references=config["references"]
        )
    return result


def get_quality_risk_description(risk_level: float) -> str:
    """Get a human-readable description of a quality risk level.
    
    Args:
        risk_level: The risk level value (0.0, 0.5, or 1.0)
    
    Returns:
        Human-readable description
    """
    if risk_level == QualityRiskLevel.LOW.value:
        return "LOW risk: Minimal quality impact, suitable for production use"
    elif risk_level == QualityRiskLevel.MEDIUM.value:
        return "MEDIUM risk: Noticeable quality impact, may be acceptable for some use cases"
    elif risk_level == QualityRiskLevel.HIGH.value:
        return "HIGH risk: Significant quality degradation, use with caution"
    else:
        return f"UNKNOWN risk level: {risk_level}"
