"""Validator for quantization result JSON data.

This module validates the structure and consistency of JSON output
from the analyze subcommand.
"""

from typing import Dict, Any, List, Optional, Union

from .constants import (
    ErrorCode,
    REQUIRED_RESULT_FIELDS,
    QuantizationFamily
)
from .datatypes import ValidationError


def validate_field_type(
    value: Any,
    expected_type: type,
    field_name: str
) -> Optional[ValidationError]:
    """Validate that a field has the expected type.
    
    Args:
        value: The value to check
        expected_type: The expected type
        field_name: Name of the field for error messages
    
    Returns:
        ValidationError if invalid, None if valid
    """
    if expected_type == float:
        if not isinstance(value, (int, float)):
            return ValidationError(
                error_code=ErrorCode.INVALID_FIELD_TYPE,
                message=f"Field '{field_name}' expected numeric (int/float), got {type(value).__name__}",
                field=field_name
            )
    elif not isinstance(value, expected_type):
        return ValidationError(
            error_code=ErrorCode.INVALID_FIELD_TYPE,
            message=f"Field '{field_name}' expected {expected_type.__name__}, got {type(value).__name__}",
            field=field_name
        )
    
    return None


def validate_result_structure(data: Dict[str, Any]) -> List[ValidationError]:
    """Validate the structure of a quantization result JSON.
    
    Checks:
    1. All required fields are present
    2. Fields have correct types
    3. Numeric values are in valid ranges
    
    Args:
        data: The parsed JSON data as dictionary
    
    Returns:
        List of ValidationError objects. Empty list means valid.
    """
    errors: List[ValidationError] = []
    
    for field_name, field_spec in REQUIRED_RESULT_FIELDS.items():
        is_required = field_spec.get("required", False)
        
        if field_name not in data:
            if is_required:
                errors.append(ValidationError(
                    error_code=ErrorCode.MISSING_FIELD,
                    message=f"Missing required field: {field_name}",
                    field=field_name
                ))
            continue
        
        value = data[field_name]
        expected_type = field_spec.get("type", str)
        
        type_error = validate_field_type(value, expected_type, field_name)
        if type_error:
            errors.append(type_error)
            continue
        
        if isinstance(value, (int, float)):
            min_value = field_spec.get("min_value")
            if min_value is not None and value < min_value:
                errors.append(ValidationError(
                    error_code=ErrorCode.INVALID_VALUE,
                    message=f"Field '{field_name}' value {value} is less than minimum {min_value}",
                    field=field_name
                ))
            
            valid_values = field_spec.get("valid_values")
            if valid_values is not None and float(value) not in valid_values:
                errors.append(ValidationError(
                    error_code=ErrorCode.INVALID_VALUE,
                    message=f"Field '{field_name}' value {value} not in valid values {valid_values}",
                    field=field_name
                ))
    
    return errors


def validate_result_consistency(data: Dict[str, Any]) -> List[ValidationError]:
    """Validate the consistency of the result data.
    
    Checks:
    1. Baseline and quant configs are valid
    2. Relative values make sense (should typically be > 0)
    
    Args:
        data: The parsed JSON data as dictionary
    
    Returns:
        List of ValidationError objects. Empty list means valid.
    """
    errors: List[ValidationError] = []
    
    baseline_config = data.get("baseline_config", {})
    quant_config = data.get("quant_config", {})
    
    if baseline_config:
        baseline_bit_width = baseline_config.get("bit_width")
        if baseline_bit_width is not None and baseline_bit_width <= 0:
            errors.append(ValidationError(
                error_code=ErrorCode.INVALID_VALUE,
                message=f"Baseline bit_width {baseline_bit_width} must be positive",
                field="baseline_config.bit_width"
            ))
    
    if quant_config:
        family_str = quant_config.get("family", "")
        valid_families = [f.value for f in QuantizationFamily]
        if family_str and family_str not in valid_families:
            errors.append(ValidationError(
                error_code=ErrorCode.INVALID_QUANT_FAMILY,
                message=f"Quantization family '{family_str}' not in valid values: {valid_families}",
                field="quant_config.family"
            ))
        
        bit_width = quant_config.get("bit_width")
        if bit_width is not None and bit_width <= 0:
            errors.append(ValidationError(
                error_code=ErrorCode.INVALID_VALUE,
                message=f"Quantization bit_width {bit_width} must be positive",
                field="quant_config.bit_width"
            ))
    
    relative_latency = data.get("relative_latency")
    if relative_latency is not None:
        if relative_latency <= 0:
            errors.append(ValidationError(
                error_code=ErrorCode.INVALID_VALUE,
                message=f"relative_latency {relative_latency} must be > 0",
                field="relative_latency"
            ))
    
    relative_memory = data.get("relative_memory")
    if relative_memory is not None:
        if relative_memory <= 0:
            errors.append(ValidationError(
                error_code=ErrorCode.INVALID_VALUE,
                message=f"relative_memory {relative_memory} must be > 0",
                field="relative_memory"
            ))
    
    return errors


def validate_quantization_result(data: Dict[str, Any]) -> List[ValidationError]:
    """Full validation of a quantization result.
    
    Combines structure validation and consistency validation.
    
    Args:
        data: The parsed JSON data as dictionary
    
    Returns:
        List of ValidationError objects. Empty list means valid.
    """
    errors: List[ValidationError] = []
    
    errors.extend(validate_result_structure(data))
    
    if not errors:
        errors.extend(validate_result_consistency(data))
    
    return errors


def is_valid_quantization_result(data: Dict[str, Any]) -> bool:
    """Check if a quantization result is valid (no errors).
    
    Args:
        data: The parsed JSON data as dictionary
    
    Returns:
        True if valid, False otherwise
    """
    return len(validate_quantization_result(data)) == 0


def validate_with_extreme_values() -> List[ValidationError]:
    """Validate handling of extreme values (for testing).
    
    This is a helper to demonstrate how the validator handles
    invalid values like 0 or negative ratios.
    
    Returns:
        List of errors for test data with extreme values
    """
    test_data_with_zeros = {
        "baseline_config": {"family": "fp16", "bit_width": 16},
        "quant_config": {"family": "int8", "bit_width": 8},
        "relative_latency": 0.0,
        "relative_memory": 0.0,
        "quality_risk_level": 0.0
    }
    
    test_data_with_negatives = {
        "baseline_config": {"family": "fp16", "bit_width": 16},
        "quant_config": {"family": "int8", "bit_width": 8},
        "relative_latency": -1.0,
        "relative_memory": -0.5,
        "quality_risk_level": 0.0
    }
    
    errors: List[ValidationError] = []
    errors.extend(validate_quantization_result(test_data_with_zeros))
    errors.extend(validate_quantization_result(test_data_with_negatives))
    
    return errors
