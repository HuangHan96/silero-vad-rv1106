# Silero VAD 6.2 RV1106 评估记录

## 环境

| 项目 | 值 |
| --- | --- |
| 板端 | RV1106 (LuckFox Pico), NPU ~1 TOPS |
| RKNN Runtime | API 2.3.2, driver 0.9.2 |
| 音频格式 | 16 kHz, mono, s16le |
| VAD 帧长 | 512 samples / 32 ms |

## 问题背景

Silero VAD 6.2 完整模型 (encoder + LSTM + decoder) 直接量化为 RV1106 RKNN 后：
- w8a8: self-test 输出 0，真实人声 frame 36 崩溃 (`rknn_run failed: -1`)
- w16a16i: self-test 输出异常 0.5，真实人声几帧后崩溃
- w16a16i_dfp: self-test 正常，真实人声 frame 14 崩溃

根因：LSTM hidden state 在真实语音中达到 ±45 的动态范围，int8 量化 (scale ~0.3) 导致 NPU 内部状态溢出。

## 解决方案：Split 模式

将模型拆分为：
1. **Encoder** (NPU int8): 4×Conv1d+ReLU, 输入 [1,129,4] STFT 特征
2. **LSTM** (CPU float32): hidden_size=128, 权重从二进制文件加载
3. **Decoder** (CPU float32): ReLU + Linear(128→1) + Sigmoid

## 板端测试结果

### Split 模式 vs ONNX 参考

| 样本 | 指标 | ONNX float32 | Split RKNN 板端 |
| --- | --- | ---: | ---: |
| narsil_1 | >0.5 / maxrun@0.5 | 291 / 173 | 292 / 174 |
| narsil_mlk | >0.5 / maxrun@0.5 | 218 / 79 | 220 / 79 |
| lei_jun_60s | >0.5 / maxrun@0.5 | 500 / 76 | 497 / 75 |
| ambient | max prob | 0.065 | 0.027 |

### 概率分布

| 样本 | frames | min | p50 | p90 | max | >0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| narsil_1 | 326 | 0.0068 | 0.9999 | 1.0000 | 1.0000 | 292 |
| narsil_mlk | 406 | 0.0014 | 0.9037 | 0.9997 | 1.0000 | 220 |
| lei_jun_60s | 1875 | 0.0000 | 0.0067 | 0.9983 | 1.0000 | 497 |
| ambient | 250 | 0.0000 | 0.0001 | 0.0027 | 0.0268 | 0 |

### 阈值 Sweep

| 阈值 | 环境音触发 | narsil_1 触发/最长连续 | narsil_mlk 触发/最长连续 | lei_jun 触发/最长连续 |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0/250, run 0 | 306/326, run 292 | 316/406, run 92 | 652/1875, run 92 |
| 0.08 | 0/250, run 0 | 302/326, run 291 | 284/406, run 88 | 611/1875, run 91 |
| 0.10 | 0/250, run 0 | 299/326, run 237 | 278/406, run 83 | 590/1875, run 88 |
| 0.50 | 0/250, run 0 | 292/326, run 174 | 220/406, run 79 | 497/1875, run 75 |

### 性能

| 指标 | 值 |
| --- | --- |
| 单帧延迟 | 21.8 ms |
| 帧周期 | 32.0 ms |
| 实时比 | 0.68x |

## 失败尝试记录

### 全量 RKNN (LSTM on NPU)

| 配置 | 板端结果 |
| --- | --- |
| w8a8 opt=1 | frame 38 崩溃 |
| w8a8 opt=0 | frame 38 崩溃 |
| w8a8 mmse | frame 53 崩溃 (ambient), frame 147 崩溃 (speech) |
| w8a8 去除 ReduceMean | 仍然崩溃 (确认是 LSTM 问题) |
| w16a16i | 输出异常 + 崩溃 |
| w16a16i_dfp | frame 14 崩溃 |
| do_quantization=False | RV1106 不支持 |
| auto_hybrid | 构建失败 (float16 Conv 不支持) |

### sherpa-onnx v4

- ONNX 效果好 (人声/噪声分离度高)
- 但 raw-audio 图含 Sqrt op，RV1106 CPU fallback 不支持，无法加载

## 结论

RV1106 LSTM int8 量化对大动态范围 hidden state 不稳定。Split 模式 (encoder NPU + LSTM CPU) 是当前唯一可行方案，效果与原模型一致。
