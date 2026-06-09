#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Конвертация акустической модели T-one fp16 -> fp32.

Зачем: модель T-one экспортирована в fp16. На CUDA onnxruntime каждый внутренний
fp32<->fp16 Cast (паттерн BatchNorm -> Cast -> QuickGelu, ~18 шт.) выкидывает на CPU,
обрамляя парой MemcpyToHost/FromHost -> 38 Memcpy-нод -> 38 синхронизаций GPU на forward.
На Pascal (GTX 1080) это держит GPU на ~62% util и делает forward ~80 мс.

Перевод весов/Cast'ов в fp32 убирает эти CPU-островки (проверено: 38 -> 2 Memcpy),
текст декода не меняется (fp32 точнее fp16). Состояние модели тоже становится fp32 —
на стороне приложения нужен FP32-сабкласс StreamingCTCModel (см. Recognizer/tone_engine.py).

Usage:
    python tools/convert_tone_fp32.py <src_model.onnx> <dst_model.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

FP16, FP32 = TensorProto.FLOAT16, TensorProto.FLOAT


def _convert_graph(g) -> None:
    """Рекурсивно (вкл. подграфы If/Loop/Scan) переводит fp16 -> fp32."""
    for init in g.initializer:
        if init.data_type == FP16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for node in g.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FP16:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32)))
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == FP16:
                    a.i = FP32  # Cast->fp16 становится Cast->fp32 (no-op после ретайпа входов)
        for a in node.attribute:
            if a.HasField("g"):
                _convert_graph(a.g)
            for sg in a.graphs:
                _convert_graph(sg)
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.type.tensor_type.elem_type == FP16:
            vi.type.tensor_type.elem_type = FP32


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    _convert_graph(m.graph)
    onnx.save(m, dst)
    print(f"fp32-модель сохранена: {dst}")


if __name__ == "__main__":
    main()
