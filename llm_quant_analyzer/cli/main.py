"""Command-line interface for LLM Quantization Analyzer.

Usage:
    python -m llm_quant_analyzer list
    python -m llm_quant_analyzer analyze --family int8 --bit-width 8
    python -m llm_quant_analyzer validate --file result.json
    python -m llm_quant_analyzer validate < result.json
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional, List, TextIO

from ..core.constants import (
    QuantizationFamily,
    QualityRiskLevel,
    ErrorCode,
    EXIT_CODE_MAP,
    GGUF_TYPE_DETAILS,
    QUANTIZATION_CONFIGS
)
from ..core.datatypes import (
    QuantizationConfig,
    BaselineConfig,
    QuantizationResult,
    ValidationError
)
from ..core.calculator import (
    calculate_quantization_impact,
    get_supported_quantizations,
    get_quality_risk_description
)
from ..core.validator import (
    validate_quantization_result
)


EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def get_exit_code(error_code: ErrorCode) -> int:
    """Get the exit code for an ErrorCode."""
    return EXIT_CODE_MAP.get(error_code, EXIT_FAILURE)


def print_error(error: ValidationError, file: TextIO = sys.stderr) -> None:
    """Print a validation error to stderr with stable error code."""
    print(error.to_string(), file=file)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="llm_quant_analyzer",
        description="LLM Quantization Analyzer - Analyze INT8, INT4, and GGUF quantization impacts",
        epilog="Quality Risk Level: 0.0=LOW (minimal impact), 0.5=MEDIUM (noticeable), 1.0=HIGH (significant)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="LLM Quantization Analyzer 0.1.0"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands"
    )
    
    list_parser = subparsers.add_parser(
        "list",
        help="List supported quantization families and bit widths",
        description="List all supported quantization configurations with human-readable summaries"
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable text"
    )
    
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze quantization impact compared to baseline",
        description="Calculate relative latency, relative memory, and quality risk level for a quantization configuration"
    )
    analyze_parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=[f.value for f in QuantizationFamily],
        help="Quantization family: int8, int4, or gguf"
    )
    analyze_parser.add_argument(
        "--bit-width",
        type=int,
        required=True,
        help="Bit width for quantization (e.g., 4, 8 for gguf: 2,3,4,5,6,8)"
    )
    analyze_parser.add_argument(
        "--baseline-family",
        type=str,
        default="fp16",
        help="Baseline precision family (default: fp16)"
    )
    analyze_parser.add_argument(
        "--baseline-bit-width",
        type=int,
        default=16,
        help="Baseline bit width (default: 16)"
    )
    
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate JSON output from analyze command",
        description="Validate the structure and consistency of quantization result JSON"
    )
    validate_parser.add_argument(
        "--file",
        type=str,
        help="Path to JSON file to validate (if not provided, reads from stdin)"
    )
    validate_parser.add_argument(
        "--help-errors",
        action="store_true",
        help="Show detailed error code information and exit codes"
    )
    
    return parser


def get_error_code_help() -> str:
    """Generate help text for error codes."""
    lines = [
        "Error Codes and Exit Codes:",
        "",
        "Error Code          | Exit Code | Description",
        "--------------------|-----------|------------------------",
    ]
    
    for error_code in ErrorCode:
        exit_code = get_exit_code(error_code)
        description = {
            ErrorCode.SUCCESS: "Operation completed successfully",
            ErrorCode.INVALID_QUANT_FAMILY: "Invalid quantization family specified",
            ErrorCode.INVALID_BIT_WIDTH: "Invalid bit width specified",
            ErrorCode.UNSUPPORTED_COMBINATION: "Unsupported family/bit-width combination",
            ErrorCode.MISSING_FIELD: "Missing required field in JSON",
            ErrorCode.INVALID_FIELD_TYPE: "Field has incorrect type",
            ErrorCode.INVALID_VALUE: "Field value is invalid (e.g., negative ratio)",
            ErrorCode.INCONSISTENT_DATA: "Data is internally inconsistent",
            ErrorCode.IO_ERROR: "I/O error reading/writing file",
            ErrorCode.PARSE_ERROR: "JSON parse error",
        }.get(error_code, "Unknown error")
        
        lines.append(f"{error_code.value:<20}| {exit_code:<10}| {description}")
    
    lines.extend([
        "",
        "Usage in scripts:",
        "  - Check exit code: $?",
        "  - Grep for error codes: | grep 'E_'"
    ])
    
    return "\n".join(lines)


def cmd_list(args: argparse.Namespace) -> int:
    """Execute the 'list' subcommand."""
    supported = get_supported_quantizations()
    
    if args.json:
        output = {}
        for family, info in supported.items():
            output[family.value] = {
                "bit_widths": info.bit_widths,
                "description": info.description,
                "typical_use_cases": info.typical_use_cases,
                "references": info.references
            }
        print(json.dumps(output, indent=2))
    else:
        for family, info in supported.items():
            print(info.to_human_readable())
    
    return EXIT_SUCCESS


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute the 'analyze' subcommand."""
    try:
        family = QuantizationFamily(args.family)
    except ValueError:
        error = ValidationError(
            error_code=ErrorCode.INVALID_QUANT_FAMILY,
            message=f"Invalid quantization family: {args.family}",
            field="family"
        )
        print_error(error)
        return get_exit_code(error.error_code)
    
    quant_config = QuantizationConfig(
        family=family,
        bit_width=args.bit_width
    )
    
    baseline_config = BaselineConfig(
        family=args.baseline_family,
        bit_width=args.baseline_bit_width
    )
    
    result, error = calculate_quantization_impact(quant_config, baseline_config)
    
    if error:
        print_error(error)
        return get_exit_code(error.error_code)
    
    if result:
        print(json.dumps(result.to_dict()))
        return EXIT_SUCCESS
    
    return EXIT_FAILURE


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the 'validate' subcommand."""
    if args.help_errors:
        print(get_error_code_help())
        return EXIT_SUCCESS
    
    json_data: Optional[Dict[str, Any]] = None
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except IOError as e:
            error = ValidationError(
                error_code=ErrorCode.IO_ERROR,
                message=f"Failed to read file: {e}",
                field=args.file
            )
            print_error(error)
            return get_exit_code(error.error_code)
        except json.JSONDecodeError as e:
            error = ValidationError(
                error_code=ErrorCode.PARSE_ERROR,
                message=f"JSON parse error: {e}",
                field=args.file
            )
            print_error(error)
            return get_exit_code(error.error_code)
    else:
        try:
            json_str = sys.stdin.read()
            json_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            error = ValidationError(
                error_code=ErrorCode.PARSE_ERROR,
                message=f"JSON parse error from stdin: {e}",
                field="stdin"
            )
            print_error(error)
            return get_exit_code(error.error_code)
    
    if json_data is None:
        error = ValidationError(
            error_code=ErrorCode.MISSING_FIELD,
            message="No JSON data provided",
            field=None
        )
        print_error(error)
        return get_exit_code(error.error_code)
    
    errors = validate_quantization_result(json_data)
    
    if errors:
        for error in errors:
            print_error(error)
        return get_exit_code(errors[0].error_code)
    
    print("ok", file=sys.stderr)
    return EXIT_SUCCESS


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "list":
        return cmd_list(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "validate":
        return cmd_validate(args)
    else:
        parser.print_help()
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
