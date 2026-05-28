#!/usr/bin/env python3
"""Split a Silero VAD 6.2 fixed ONNX into encoder-only ONNX + LSTM/decoder weights binary.

Usage:
    python split_encoder.py \
        --onnx silero_vad_6_2_fixed_ctx64_simplified.onnx \
        --encoder-output encoder_only.onnx \
        --weights-output lstm_decoder_weights.bin
"""

from __future__ import annotations

import argparse
import copy
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="input fixed ONNX (STFT externalized)")
    parser.add_argument("--encoder-output", required=True, help="output encoder-only ONNX")
    parser.add_argument("--weights-output", required=True, help="output LSTM+decoder weights binary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    m = onnx.load(args.onnx)

    # --- Encoder-only ONNX (nodes 0-7: 4x Conv+ReLU) ---
    encoder = onnx.ModelProto()
    encoder.CopyFrom(m)
    del encoder.graph.node[:]
    del encoder.graph.output[:]
    del encoder.graph.initializer[:]

    for i in range(8):
        encoder.graph.node.append(copy.deepcopy(m.graph.node[i]))

    # Find encoder output name (last ReLU output)
    encoder_output_name = m.graph.node[7].output[0]

    # Rename to 'encoder_out'
    for n in encoder.graph.node:
        for i, o in enumerate(n.output):
            if o == encoder_output_name:
                n.output[i] = "encoder_out"

    encoder_out_vi = helper.make_tensor_value_info("encoder_out", TensorProto.FLOAT, [1, 128, 1])
    encoder.graph.output.append(encoder_out_vi)

    # Keep only 'input' graph input
    kept_inputs = [copy.deepcopy(v) for v in m.graph.input if v.name == "input"]
    del encoder.graph.input[:]
    encoder.graph.input.extend(kept_inputs)

    # Copy needed initializers
    used_names = set()
    for n in encoder.graph.node:
        used_names.update(inp for inp in n.input if inp)
    for init in m.graph.initializer:
        if init.name in used_names:
            encoder.graph.initializer.append(copy.deepcopy(init))

    onnx.checker.check_model(encoder)
    Path(args.encoder_output).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(encoder, args.encoder_output)
    print(f"Encoder ONNX: {args.encoder_output} ({len(encoder.graph.node)} nodes)")

    # --- Extract LSTM + decoder weights ---
    lstm_node = next(n for n in m.graph.node if n.op_type == "LSTM")
    W = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == lstm_node.input[1]))
    R = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == lstm_node.input[2]))
    B = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == lstm_node.input[3]))

    decoder_conv = next(n for n in m.graph.node if n.op_type == "Conv" and "decoder" in n.name)
    dec_W = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == decoder_conv.input[1]))
    dec_B = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == decoder_conv.input[2]))

    hidden_size = 128
    Path(args.weights_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.weights_output, "wb") as f:
        f.write(b"SVLW")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", hidden_size))
        f.write(struct.pack("<I", hidden_size))  # input_size
        f.write(W.astype(np.float32).tobytes())
        f.write(R.astype(np.float32).tobytes())
        f.write(B.astype(np.float32).tobytes())
        f.write(dec_W.reshape(-1).astype(np.float32).tobytes())
        f.write(dec_B.reshape(-1).astype(np.float32).tobytes())

    import os
    size = os.path.getsize(args.weights_output)
    print(f"Weights binary: {args.weights_output} ({size} bytes)")
    print(f"  LSTM W: {W.shape}, R: {R.shape}, B: {B.shape}")
    print(f"  Decoder weight: {dec_W.shape}, bias: {dec_B.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
