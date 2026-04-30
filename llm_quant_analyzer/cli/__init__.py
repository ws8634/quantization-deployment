"""CLI layer for LLM Quantization Analyzer.

This module contains the command-line interface with three subcommands:
1. list - List supported quantization families and bit widths
2. analyze - Analyze quantization impact compared to baseline
3. validate - Validate JSON output from analyze command
"""

from . import main

__all__ = ["main"]
