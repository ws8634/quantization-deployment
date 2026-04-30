"""Unit tests for the core logic layer.

Tests cover:
1. Unsupported bit width combinations
2. JSON validation with missing fields
3. Extreme input values (0 or negative ratios)
"""

import unittest
from typing import Dict, Any

from llm_quant_analyzer.core.constants import (
    QuantizationFamily,
    QualityRiskLevel,
    ErrorCode
)
from llm_quant_analyzer.core.datatypes import (
    QuantizationConfig,
    BaselineConfig,
    QuantizationResult,
    ValidationError
)
from llm_quant_analyzer.core.calculator import (
    is_bit_width_supported,
    validate_quantization_config,
    calculate_quantization_impact,
    get_supported_quantizations
)
from llm_quant_analyzer.core.validator import (
    validate_quantization_result,
    validate_result_structure,
    validate_result_consistency,
    is_valid_quantization_result
)


class TestBitWidthSupport(unittest.TestCase):
    """Test bit width support validation.
    
    Boundary case 1: Unsupported bit width combinations
    """
    
    def test_int8_supports_only_8_bits(self) -> None:
        """INT8 should only support 8-bit width."""
        self.assertTrue(is_bit_width_supported(QuantizationFamily.INT8, 8))
        self.assertFalse(is_bit_width_supported(QuantizationFamily.INT8, 4))
        self.assertFalse(is_bit_width_supported(QuantizationFamily.INT8, 16))
        self.assertFalse(is_bit_width_supported(QuantizationFamily.INT8, 0))
        self.assertFalse(is_bit_width_supported(QuantizationFamily.INT8, -8))
    
    def test_int4_supports_only_4_bits(self) -> None:
        """INT4 should only support 4-bit width."""
        self.assertTrue(is_bit_width_supported(QuantizationFamily.INT4, 4))
        self.assertFalse(is_bit_width_supported(QuantizationFamily.INT4, 8))
        self.assertFalse(is_bit_width_supported(QuantizationFamily.INT4, 2))
    
    def test_gguf_supports_multiple_bit_widths(self) -> None:
        """GGUF should support multiple bit widths."""
        for bit_width in [2, 3, 4, 5, 6, 8]:
            with self.subTest(bit_width=bit_width):
                self.assertTrue(is_bit_width_supported(QuantizationFamily.GGUF, bit_width))
        
        self.assertFalse(is_bit_width_supported(QuantizationFamily.GGUF, 1))
        self.assertFalse(is_bit_width_supported(QuantizationFamily.GGUF, 7))
        self.assertFalse(is_bit_width_supported(QuantizationFamily.GGUF, 16))
    
    def test_validate_unsupported_bit_width_returns_error(self) -> None:
        """Validation should return error for unsupported bit width combinations."""
        config = QuantizationConfig(
            family=QuantizationFamily.INT8,
            bit_width=4
        )
        error = validate_quantization_config(config)
        
        self.assertIsNotNone(error)
        if error:
            self.assertEqual(error.error_code, ErrorCode.UNSUPPORTED_COMBINATION)
            self.assertIn("not supported", error.message.lower())
            self.assertEqual(error.field, "bit_width")
    
    def test_calculate_with_unsupported_combination_fails(self) -> None:
        """Calculation should fail with unsupported bit width combinations."""
        quant_config = QuantizationConfig(
            family=QuantizationFamily.INT4,
            bit_width=8
        )
        
        result, error = calculate_quantization_impact(quant_config)
        
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        if error:
            self.assertEqual(error.error_code, ErrorCode.UNSUPPORTED_COMBINATION)
    
    def test_supported_configs_pass_validation(self) -> None:
        """Valid configurations should pass validation."""
        valid_configs = [
            QuantizationConfig(family=QuantizationFamily.INT8, bit_width=8),
            QuantizationConfig(family=QuantizationFamily.INT4, bit_width=4),
            QuantizationConfig(family=QuantizationFamily.GGUF, bit_width=4),
            QuantizationConfig(family=QuantizationFamily.GGUF, bit_width=8),
        ]
        
        for config in valid_configs:
            with self.subTest(config=f"{config.family.value}-{config.bit_width}"):
                error = validate_quantization_config(config)
                self.assertIsNone(error)


class TestJSONValidation(unittest.TestCase):
    """Test JSON validation with missing fields.
    
    Boundary case 2: Missing fields in JSON validation
    """
    
    def create_valid_result(self) -> Dict[str, Any]:
        """Create a valid result dictionary for testing."""
        return {
            "baseline_config": {
                "family": "fp16",
                "bit_width": 16,
                "description": "FP16 baseline"
            },
            "quant_config": {
                "family": "int8",
                "bit_width": 8,
                "custom_params": {}
            },
            "relative_latency": 0.7,
            "relative_memory": 0.25,
            "quality_risk_level": 0.0,
            "metadata": {
                "reference": "Test Reference"
            }
        }
    
    def test_valid_result_passes_validation(self) -> None:
        """A valid result should pass validation."""
        valid_data = self.create_valid_result()
        errors = validate_quantization_result(valid_data)
        
        self.assertEqual(len(errors), 0)
        self.assertTrue(is_valid_quantization_result(valid_data))
    
    def test_missing_baseline_config_fails(self) -> None:
        """Missing baseline_config should fail validation."""
        data = self.create_valid_result()
        del data["baseline_config"]
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
        missing_field_errors = [
            e for e in errors if e.error_code == ErrorCode.MISSING_FIELD
        ]
        self.assertGreater(len(missing_field_errors), 0)
        
        baseline_errors = [
            e for e in missing_field_errors 
            if e.field and "baseline" in e.field
        ]
        self.assertGreater(len(baseline_errors), 0)
    
    def test_missing_quant_config_fails(self) -> None:
        """Missing quant_config should fail validation."""
        data = self.create_valid_result()
        del data["quant_config"]
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
        missing_field_errors = [
            e for e in errors if e.error_code == ErrorCode.MISSING_FIELD
        ]
        self.assertGreater(len(missing_field_errors), 0)
    
    def test_missing_relative_latency_fails(self) -> None:
        """Missing relative_latency should fail validation."""
        data = self.create_valid_result()
        del data["relative_latency"]
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
        missing_field_errors = [
            e for e in errors if e.error_code == ErrorCode.MISSING_FIELD
        ]
        self.assertGreater(len(missing_field_errors), 0)
    
    def test_missing_relative_memory_fails(self) -> None:
        """Missing relative_memory should fail validation."""
        data = self.create_valid_result()
        del data["relative_memory"]
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
    
    def test_missing_quality_risk_level_fails(self) -> None:
        """Missing quality_risk_level should fail validation."""
        data = self.create_valid_result()
        del data["quality_risk_level"]
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
    
    def test_missing_metadata_is_optional(self) -> None:
        """Missing metadata should be optional and pass validation."""
        data = self.create_valid_result()
        del data["metadata"]
        
        errors = validate_quantization_result(data)
        
        self.assertEqual(len(errors), 0)
        self.assertTrue(is_valid_quantization_result(data))
    
    def test_multiple_missing_fields_reported(self) -> None:
        """Multiple missing fields should all be reported."""
        data = self.create_valid_result()
        del data["baseline_config"]
        del data["quant_config"]
        del data["relative_latency"]
        
        errors = validate_quantization_result(data)
        
        missing_field_errors = [
            e for e in errors if e.error_code == ErrorCode.MISSING_FIELD
        ]
        self.assertGreaterEqual(len(missing_field_errors), 3)


class TestExtremeValues(unittest.TestCase):
    """Test handling of extreme input values.
    
    Boundary case 3: Extreme input values (0 or negative ratios)
    """
    
    def create_valid_result(self) -> Dict[str, Any]:
        """Create a valid result dictionary for testing."""
        return {
            "baseline_config": {"family": "fp16", "bit_width": 16},
            "quant_config": {"family": "int8", "bit_width": 8},
            "relative_latency": 0.7,
            "relative_memory": 0.25,
            "quality_risk_level": 0.0
        }
    
    def test_relative_latency_zero_fails(self) -> None:
        """relative_latency of 0 should fail validation."""
        data = self.create_valid_result()
        data["relative_latency"] = 0.0
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
        invalid_value_errors = [
            e for e in errors if e.error_code == ErrorCode.INVALID_VALUE
        ]
        self.assertGreater(len(invalid_value_errors), 0)
        
        latency_errors = [
            e for e in invalid_value_errors
            if e.field and "latency" in e.field
        ]
        self.assertGreater(len(latency_errors), 0)
    
    def test_relative_latency_negative_fails(self) -> None:
        """Negative relative_latency should fail validation."""
        data = self.create_valid_result()
        data["relative_latency"] = -0.5
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
        invalid_value_errors = [
            e for e in errors if e.error_code == ErrorCode.INVALID_VALUE
        ]
        self.assertGreater(len(invalid_value_errors), 0)
    
    def test_relative_memory_zero_fails(self) -> None:
        """relative_memory of 0 should fail validation."""
        data = self.create_valid_result()
        data["relative_memory"] = 0.0
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
    
    def test_relative_memory_negative_fails(self) -> None:
        """Negative relative_memory should fail validation."""
        data = self.create_valid_result()
        data["relative_memory"] = -1.0
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
    
    def test_invalid_quality_risk_level_fails(self) -> None:
        """Invalid quality_risk_level values should fail validation."""
        invalid_values = [-1.0, 0.25, 0.75, 2.0]
        
        for value in invalid_values:
            with self.subTest(value=value):
                data = self.create_valid_result()
                data["quality_risk_level"] = value
                
                errors = validate_quantization_result(data)
                
                self.assertGreater(len(errors), 0)
                invalid_value_errors = [
                    e for e in errors if e.error_code == ErrorCode.INVALID_VALUE
                ]
                self.assertGreater(len(invalid_value_errors), 0)
    
    def test_valid_quality_risk_level_passes(self) -> None:
        """Valid quality_risk_level values should pass validation."""
        valid_values = [0.0, 0.5, 1.0]
        
        for value in valid_values:
            with self.subTest(value=value):
                data = self.create_valid_result()
                data["quality_risk_level"] = value
                
                errors = validate_quantization_result(data)
                
                self.assertEqual(len(errors), 0)
    
    def test_negative_bit_width_in_config_fails(self) -> None:
        """Negative bit_width in config should fail consistency validation."""
        data = self.create_valid_result()
        data["quant_config"]["bit_width"] = -4
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
    
    def test_zero_bit_width_in_config_fails(self) -> None:
        """Zero bit_width in config should fail consistency validation."""
        data = self.create_valid_result()
        data["baseline_config"]["bit_width"] = 0
        
        errors = validate_quantization_result(data)
        
        self.assertGreater(len(errors), 0)
    
    def test_field_type_validation(self) -> None:
        """Incorrect field types should fail validation."""
        test_cases = [
            ("relative_latency", "not_a_number"),
            ("relative_memory", []),
            ("quality_risk_level", "high"),
            ("baseline_config", "should_be_dict"),
            ("quant_config", 123),
        ]
        
        for field_name, invalid_value in test_cases:
            with self.subTest(field=field_name):
                data = self.create_valid_result()
                data[field_name] = invalid_value
                
                errors = validate_quantization_result(data)
                
                self.assertGreater(len(errors), 0)
                type_errors = [
                    e for e in errors 
                    if e.error_code == ErrorCode.INVALID_FIELD_TYPE
                ]
                self.assertGreater(len(type_errors), 0)


class TestCalculatorFunctionality(unittest.TestCase):
    """Test the calculator functionality with valid inputs."""
    
    def test_int8_calculation(self) -> None:
        """INT8 quantization should return expected ratios."""
        config = QuantizationConfig(
            family=QuantizationFamily.INT8,
            bit_width=8
        )
        
        result, error = calculate_quantization_impact(config)
        
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        
        if result:
            self.assertEqual(result.relative_latency, 0.7)
            self.assertEqual(result.relative_memory, 0.25)
            self.assertEqual(result.quality_risk_level, QualityRiskLevel.LOW.value)
    
    def test_int4_calculation(self) -> None:
        """INT4 quantization should return expected ratios."""
        config = QuantizationConfig(
            family=QuantizationFamily.INT4,
            bit_width=4
        )
        
        result, error = calculate_quantization_impact(config)
        
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        
        if result:
            self.assertEqual(result.relative_latency, 0.5)
            self.assertEqual(result.relative_memory, 0.125)
            self.assertEqual(result.quality_risk_level, QualityRiskLevel.MEDIUM.value)
    
    def test_gguf_q4_km_calculation(self) -> None:
        """GGUF Q4_K_M (4-bit) should return expected ratios."""
        config = QuantizationConfig(
            family=QuantizationFamily.GGUF,
            bit_width=4
        )
        
        result, error = calculate_quantization_impact(config)
        
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        
        if result:
            self.assertEqual(result.relative_latency, 0.5)
            self.assertEqual(result.relative_memory, 0.1875)
            self.assertEqual(result.quality_risk_level, QualityRiskLevel.LOW.value)
    
    def test_gguf_q2_k_high_risk(self) -> None:
        """GGUF Q2_K (2-bit) should have HIGH quality risk."""
        config = QuantizationConfig(
            family=QuantizationFamily.GGUF,
            bit_width=2
        )
        
        result, error = calculate_quantization_impact(config)
        
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        
        if result:
            self.assertEqual(result.quality_risk_level, QualityRiskLevel.HIGH.value)
    
    def test_custom_baseline_config(self) -> None:
        """Custom baseline config should be included in result."""
        quant_config = QuantizationConfig(
            family=QuantizationFamily.INT8,
            bit_width=8
        )
        baseline_config = BaselineConfig(
            family="fp32",
            bit_width=32,
            description="Custom FP32 baseline"
        )
        
        result, error = calculate_quantization_impact(quant_config, baseline_config)
        
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        
        if result:
            self.assertEqual(result.baseline_config["family"], "fp32")
            self.assertEqual(result.baseline_config["bit_width"], 32)
    
    def test_result_to_dict_round_trip(self) -> None:
        """Result should survive to_dict and from_dict round trip."""
        config = QuantizationConfig(
            family=QuantizationFamily.INT8,
            bit_width=8
        )
        
        result, error = calculate_quantization_impact(config)
        
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        
        if result:
            result_dict = result.to_dict()
            
            reloaded = QuantizationResult.from_dict(result_dict)
            
            self.assertEqual(reloaded.relative_latency, result.relative_latency)
            self.assertEqual(reloaded.relative_memory, result.relative_memory)
            self.assertEqual(reloaded.quality_risk_level, result.quality_risk_level)


class TestSupportedQuantizations(unittest.TestCase):
    """Test get_supported_quantizations function."""
    
    def test_returns_all_families(self) -> None:
        """Should return all three quantization families."""
        supported = get_supported_quantizations()
        
        self.assertIn(QuantizationFamily.INT8, supported)
        self.assertIn(QuantizationFamily.INT4, supported)
        self.assertIn(QuantizationFamily.GGUF, supported)
    
    def test_each_has_bit_widths(self) -> None:
        """Each family should have bit widths defined."""
        supported = get_supported_quantizations()
        
        for family, info in supported.items():
            with self.subTest(family=family.value):
                self.assertGreater(len(info.bit_widths), 0)
                self.assertIsInstance(info.description, str)
                self.assertGreater(len(info.typical_use_cases), 0)
                self.assertGreater(len(info.references), 0)
    
    def test_human_readable_output(self) -> None:
        """Human-readable output should be non-empty."""
        supported = get_supported_quantizations()
        
        for family, info in supported.items():
            with self.subTest(family=family.value):
                output = info.to_human_readable()
                self.assertGreater(len(output), 0)
                self.assertIn(family.value.upper(), output)


if __name__ == "__main__":
    unittest.main()
