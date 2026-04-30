"""Data types and structures for quantization analysis."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from .constants import (
    QuantizationFamily,
    QualityRiskLevel,
    ErrorCode
)


@dataclass
class QuantizationConfig:
    """Configuration for a quantization setup.
    
    Attributes:
        family: The quantization family (INT8, INT4, GGUF)
        bit_width: The bit width for quantization
        custom_params: Optional custom parameters for advanced configurations
    """
    
    family: QuantizationFamily
    bit_width: int
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "family": self.family.value,
            "bit_width": self.bit_width,
            "custom_params": self.custom_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationConfig':
        """Create from dictionary representation."""
        family_str = data.get("family", "").lower()
        try:
            family = QuantizationFamily(family_str)
        except ValueError:
            raise ValueError(f"Invalid quantization family: {family_str}")
        
        return cls(
            family=family,
            bit_width=data.get("bit_width", 0),
            custom_params=data.get("custom_params", {})
        )


@dataclass
class BaselineConfig:
    """Configuration for the baseline reference.
    
    Typically FP16 or FP32 precision.
    """
    
    family: str = "fp16"
    bit_width: int = 16
    description: str = "FP16 baseline reference"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "family": self.family,
            "bit_width": self.bit_width,
            "description": self.description
        }


@dataclass
class QuantizationResult:
    """Result of quantization analysis comparison.
    
    Attributes:
        baseline_config: The baseline configuration (FP16, etc.)
        quant_config: The quantization configuration
        relative_latency: Latency ratio compared to baseline (< 1 means faster)
        relative_memory: Memory ratio compared to baseline (< 1 means smaller)
        quality_risk_level: Quality risk level (0.0=LOW, 0.5=MEDIUM, 1.0=HIGH)
        metadata: Additional metadata including references
    """
    
    baseline_config: Dict[str, Any]
    quant_config: Dict[str, Any]
    relative_latency: float
    relative_memory: float
    quality_risk_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for JSON output."""
        return {
            "baseline_config": self.baseline_config,
            "quant_config": self.quant_config,
            "relative_latency": self.relative_latency,
            "relative_memory": self.relative_memory,
            "quality_risk_level": self.quality_risk_level,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationResult':
        """Create from dictionary representation."""
        return cls(
            baseline_config=data.get("baseline_config", {}),
            quant_config=data.get("quant_config", {}),
            relative_latency=data.get("relative_latency", 0.0),
            relative_memory=data.get("relative_memory", 0.0),
            quality_risk_level=data.get("quality_risk_level", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class ValidationError:
    """Represents a validation error with stable error code.
    
    Attributes:
        error_code: Stable error code enum
        message: Human-readable error message
        field: Optional field name where error occurred
    """
    
    error_code: ErrorCode
    message: str
    field: Optional[str] = None
    
    def to_string(self) -> str:
        """Format as string for stderr output."""
        if self.field:
            return f"{self.error_code.value}: {self.message} (field: {self.field})"
        return f"{self.error_code.value}: {self.message}"


@dataclass
class SupportedQuantizationInfo:
    """Information about a supported quantization type.
    
    Used for the 'list' subcommand output.
    """
    
    family: QuantizationFamily
    bit_widths: List[int]
    description: str
    typical_use_cases: List[str]
    references: List[str]
    
    def to_human_readable(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Quantization Family: {self.family.value.upper()}",
            f"  Bit Widths: {', '.join(str(b) for b in self.bit_widths)}",
            f"  Description: {self.description}",
            f"  Typical Use Cases:",
        ]
        for use_case in self.typical_use_cases:
            lines.append(f"    - {use_case}")
        lines.append(f"  References:")
        for ref in self.references:
            lines.append(f"    - {ref}")
        lines.append("")
        return "\n".join(lines)
