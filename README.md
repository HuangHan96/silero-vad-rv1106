# Silero VAD 6.2 — RV1106 Split-Mode Deployment

Silero VAD 6.2 在 RK RV1106 (1 TOPS NPU) 上的完整部署方案。

## 方案概述

RV1106 LSTM int8 量化后 hidden state 溢出导致 NPU 崩溃，因此采用 **split 模式**：

```
PCM 16kHz s16le (512 samples / 32ms)
    ↓ + 64-sample context + reflection pad
STFT magnitude [1, 129, 4]          ← C++ helper
    ↓
Encoder (4×Conv1d+ReLU)             ← NPU int8 RKNN
    ↓ [1, 128, 1] → squeeze → [128]
LSTM cell (hidden=128)              ← CPU float32
    ↓ h' [128]
ReLU + Linear(128→1) + Sigmoid      ← CPU float32
    ↓
probability ∈ [0, 1]
```

## 目录结构

```
├── README.md                               # 本文档
├── CMakeLists.txt                          # 交叉编译 rknn_vad helper
├── toolchain-arm-rockchip830.cmake         # ARM 交叉编译工具链
├── build.sh                                # 一键编译 (Linux 原生 / Docker 容器)
├── src/
│   ├── rknn_vad.cpp                        # C++ helper (支持 --weights split 模式)
│   └── rknn_api_minimal.h                  # RKNN API 头文件
├── scripts/
│   ├── export_silero_vad_v6_2_onnx.py      # JIT → ONNX 导出
│   ├── convert_silero_vad_to_rknn.py       # ONNX → RKNN (含 STFT 外置)
│   ├── split_encoder.py                    # 拆分 encoder + 提取 LSTM/decoder 权重
│   └── make_vad_calibration_dataset.py     # 真实音频校准集生成
├── models/
│   ├── silero_vad_6_2.jit                  # 源 PyTorch JIT 模型
│   ├── silero_vad_6_2_encoder_rv1106_w8a8_v1.rknn  # 部署: encoder RKNN (137KB)
│   └── silero_vad_6_2_lstm_decoder_weights.bin     # 部署: LSTM+decoder 权重 (517KB)
├── docs/
│   └── evaluation.md                       # 板端测试评估记录
└── third_party/
    └── luckfox-pico/                       # git submodule, 含工具链 + librknnmrt.so
```

## 快速开始

### 0. 初始化子模块 (首次)

```bash
git submodule update --init --recursive third_party/luckfox-pico
```

子模块包含 `arm-rockchip830-linux-uclibcgnueabihf` 工具链 (Linux x86_64 ELF) 和 `librknnmrt.so`。

### 1. 交叉编译 helper

```bash
./build.sh
```

- Linux x86_64: 原生构建
- macOS / 其他: 自动进入 Linux x86_64 容器构建

产物: `build/bin/rknn_vad` (ARM 32-bit ELF, ~120KB)

### 2. 部署到板端

```bash
scp build/bin/rknn_vad root@<board>:/oem/usr/bin/rknn_vad
scp models/silero_vad_6_2_encoder_rv1106_w8a8_v1.rknn root@<board>:/userdata/agent/
scp models/silero_vad_6_2_lstm_decoder_weights.bin root@<board>:/userdata/agent/
```

### 3. 板端运行

```bash
# Self-test (应输出接近 0 的概率)
rknn_vad --model /userdata/agent/silero_vad_6_2_encoder_rv1106_w8a8_v1.rknn \
         --weights /userdata/agent/silero_vad_6_2_lstm_decoder_weights.bin \
         --self-test

# 流式推理 (二进制协议: F + 1024字节 PCM → P <prob>)
rknn_vad --model /userdata/agent/silero_vad_6_2_encoder_rv1106_w8a8_v1.rknn \
         --weights /userdata/agent/silero_vad_6_2_lstm_decoder_weights.bin
```

## 模型转换流程 (可复现)

需要 Python 3.10+ 和：`pip install onnx onnxruntime onnxsim numpy torch rknn-toolkit2`

### Step 1: JIT → ONNX

```bash
python scripts/export_silero_vad_v6_2_onnx.py \
  --jit models/silero_vad_6_2.jit \
  --output /tmp/silero_vad_6_2_internal.onnx \
  --context-samples 64
```

### Step 2: STFT 外置 + ONNX 简化

```bash
python scripts/convert_silero_vad_to_rknn.py \
  --onnx /tmp/silero_vad_6_2_internal.onnx \
  --context-samples 64 \
  --simplify-fixed-onnx \
  --dump-fixed-onnx-only \
  --fixed-onnx-output /tmp/silero_vad_6_2_fixed.onnx
```

### Step 3: 拆分 encoder + 提取 LSTM/decoder 权重

```bash
python scripts/split_encoder.py \
  --onnx /tmp/silero_vad_6_2_fixed.onnx \
  --encoder-output /tmp/encoder_only.onnx \
  --weights-output models/silero_vad_6_2_lstm_decoder_weights.bin
```

### Step 4: 校准集生成

需要 16 kHz s16le 单声道 raw PCM 文件 (建议含人声、环境音、静音多种场景):

```bash
python scripts/make_vad_calibration_dataset.py \
  --onnx /tmp/silero_vad_6_2_fixed.onnx \
  --output-dir /tmp/calib_v62 \
  --input audio_samples/speech.raw \
  --input audio_samples/ambient.raw \
  --context-samples 64 \
  --max-frames 3000 \
  --replace
```

### Step 5: Encoder RKNN 量化构建

```bash
python scripts/convert_silero_vad_to_rknn.py \
  --onnx /tmp/encoder_only.onnx \
  --output models/silero_vad_6_2_encoder_rv1106_w8a8_v1.rknn \
  --context-samples 64 \
  --dataset /tmp/calib_v62/dataset.txt \
  --generic-onnx
```

## 板端测试结果

| 样本 | frames | min | p50 | p90 | max | >0.5 | maxrun@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| narsil_1 (speech) | 326 | 0.007 | 1.000 | 1.000 | 1.000 | 292 | 174 |
| narsil_mlk (speech) | 406 | 0.001 | 0.904 | 1.000 | 1.000 | 220 | 79 |
| lei_jun_60s (speech) | 1875 | 0.000 | 0.007 | 0.998 | 1.000 | 497 | 75 |
| ambient (noise) | 250 | 0.000 | 0.000 | 0.003 | 0.027 | 0 | 0 |

与 ONNX float32 参考误差 < 1%。单帧延迟 21.8ms，满足 32ms 实时约束。

## 阈值建议

- 环境音最大值 0.027，阈值 **0.05** 即可完全压住
- 配合连续命中 ≥ 3 帧 (96ms) 作为 VAD 触发条件
- 人声 narsil_1 在阈值 0.05 时最长连续命中 292 帧 (9.3s)

## 性能

| 指标 | 值 |
| --- | --- |
| 单帧延迟 | 21.8 ms |
| 帧周期 | 32.0 ms |
| 实时比 | 0.68x |
| 模型总大小 | 654 KB |
| NPU 利用 | encoder only (4×Conv) |
| CPU 负载 | LSTM + decoder (float32) |

## 环境要求

| 项目 | 值 |
| --- | --- |
| 板端 | RV1106 (LuckFox Pico) |
| RKNN Runtime | API 2.3.2, driver 0.9.2 |
| 交叉编译 | arm-rockchip830-linux-uclibcgnueabihf |
| Docker | luckfoxtech/luckfox_pico:1.0 |
| Python (转换) | 3.10+, onnx, onnxruntime, onnxsim, rknn-toolkit2 |
