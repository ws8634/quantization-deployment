# LLM Quantization Analyzer

一个用于分析大语言模型量化影响的工具链，聚焦于 INT8、INT4 和 GGUF 量化方案。

## 功能概述

该工具基于公开文献和工程常识，提供量化对以下方面影响的可计算假说：
- **推理速度** (相对延迟)
- **资源占用** (相对显存/内存)
- **输出质量** (质量风险等级)

### 支持的量化类型

| 量化族 | 位宽 | 典型用途 |
|--------|------|----------|
| INT8 | 8 | 通用推理，速度/质量平衡 |
| INT4 | 4 | 内存受限环境，边缘设备 |
| GGUF | 2, 3, 4, 5, 6, 8 | llama.cpp 生态，CPU/消费级GPU |

### 质量风险等级

- **0.0 (LOW)**: 质量影响极小，适合生产环境
- **0.5 (MEDIUM)**: 质量影响明显，部分场景可接受
- **1.0 (HIGH)**: 质量退化显著，谨慎使用

## 安装

本工具无第三方依赖，仅需要 Python 3.7+。

```bash
# 克隆或解压项目后，直接使用
cd 14-llm-quantization-deployment
```

## 使用方法

### 1. 列出支持的量化类型

```bash
python -m llm_quant_analyzer list
```

使用 JSON 格式输出：
```bash
python -m llm_quant_analyzer list --json
```

### 2. 分析量化影响

```bash
# 分析 INT8 量化
python -m llm_quant_analyzer analyze --family int8 --bit-width 8

# 分析 INT4 量化
python -m llm_quant_analyzer analyze --family int4 --bit-width 4

# 分析 GGUF Q4_K_M (推荐默认)
python -m llm_quant_analyzer analyze --family gguf --bit-width 4

# 分析 GGUF Q2_K (最高压缩，高风险)
python -m llm_quant_analyzer analyze --family gguf --bit-width 2

# 使用自定义基线 (默认为 FP16)
python -m llm_quant_analyzer analyze --family int8 --bit-width 8 \
    --baseline-family fp32 --baseline-bit-width 32
```

输出示例：
```json
{
    "baseline_config": {"family": "fp16", "bit_width": 16, "description": "FP16 baseline reference"},
    "quant_config": {"family": "int8", "bit_width": 8, "custom_params": {}},
    "relative_latency": 0.7,
    "relative_memory": 0.25,
    "quality_risk_level": 0.0,
    "metadata": {
        "reference": "NVIDIA TensorRT-LLM: INT8 vs FP16 benchmarks",
        "calculation_notes": {
            "relative_latency": "< 1.0 means faster than baseline",
            "relative_memory": "< 1.0 means smaller memory footprint than baseline",
            "quality_risk_level": "0.0=LOW, 0.5=MEDIUM, 1.0=HIGH risk of quality degradation"
        }
    }
}
```

### 3. 验证 JSON 结果

```bash
# 从文件验证
python -m llm_quant_analyzer validate --file result.json

# 从标准输入验证
python -m llm_quant_analyzer analyze --family int8 --bit-width 8 | python -m llm_quant_analyzer validate

# 查看错误码说明
python -m llm_quant_analyzer validate --help-errors
```

验证成功时输出 `ok` 到 stderr，退出码为 0。

验证失败时输出带有稳定错误码的信息到 stderr，退出码非零。

## 数据来源

所有数值基于以下公开资料：

- **论文**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- **论文**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
- **文档**: llama.cpp GGUF Format Specification v3
- **厂商**: NVIDIA TensorRT-LLM, Intel Neural Compressor 文档

### 典型比率 (相对于 FP16 基线)

| 量化类型 | 相对延迟 | 相对内存 | 质量风险 |
|----------|----------|----------|----------|
| INT8 | 0.7x | 0.25x | LOW |
| INT4 | 0.5x | 0.125x | MEDIUM |
| GGUF Q2_K | 0.4x | 0.09375x | HIGH |
| GGUF Q3_K_M | 0.45x | 0.15625x | MEDIUM |
| GGUF Q4_K_M | 0.5x | 0.1875x | LOW |
| GGUF Q5_K_M | 0.6x | 0.21875x | LOW |
| GGUF Q6_K | 0.65x | 0.28125x | LOW |
| GGUF Q8_0 | 0.8x | 0.5x | LOW |

## 运行测试

```bash
# 运行所有单元测试
python -m unittest discover tests -v

# 或使用 pytest (如已安装)
pytest tests/ -v
```

测试覆盖三个边界情况：
1. 不支持的位宽组合
2. 缺少字段的 JSON 校验
3. 极端输入值 (0 或负数比率)

## 项目结构

```
14-llm-quantization-deployment/
├── llm_quant_analyzer/          # 主包
│   ├── __init__.py
│   ├── __main__.py              # python -m 入口
│   ├── core/                     # 核心逻辑层 (纯函数，无I/O)
│   │   ├── __init__.py
│   │   ├── constants.py          # 常量、枚举、配置
│   │   ├── datatypes.py          # 数据结构定义
│   │   ├── calculator.py         # 量化影响计算
│   │   └── validator.py          # JSON 验证
│   └── cli/                      # CLI 适配层
│       ├── __init__.py
│       └── main.py               # 命令行接口
├── tests/                        # 单元测试
│   ├── __init__.py
│   └── test_core.py              # 核心逻辑测试
├── examples/                     # 示例文件
│   └── sample_result.json        # 示例输出
├── README.md
└── QUICKSTART.txt                # 快速开始命令序列
```

## 错误码

使用 `python -m llm_quant_analyzer validate --help-errors` 查看完整列表。

| 错误码 | 退出码 | 说明 |
|--------|--------|------|
| OK | 0 | 成功 |
| E_INVALID_FAMILY | 1 | 无效的量化族 |
| E_INVALID_BITWIDTH | 2 | 无效的位宽 |
| E_UNSUPPORTED_COMBINATION | 3 | 不支持的组合 |
| E_MISSING_FIELD | 4 | 缺少必填字段 |
| E_INVALID_TYPE | 5 | 字段类型错误 |
| E_INVALID_VALUE | 6 | 字段值无效 |
| E_INCONSISTENT_DATA | 7 | 数据不一致 |
| E_IO_ERROR | 8 | I/O 错误 |
| E_PARSE_ERROR | 9 | JSON 解析错误 |

## 脚本集成示例

```bash
#!/bin/bash

# 分析 INT8 量化并保存结果
RESULT=$(python -m llm_quant_analyzer analyze --family int8 --bit-width 8)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "分析失败"
    exit 1
fi

# 验证结果
echo "$RESULT" | python -m llm_quant_analyzer validate 2>&1

if [ $? -eq 0 ]; then
    echo "验证通过"
else
    echo "验证失败"
    exit 1
fi

# 提取相对内存使用 (需要 jq)
RELATIVE_MEM=$(echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin)['relative_memory'])")
echo "相对内存占用: $RELATIVE_MEM"
```
