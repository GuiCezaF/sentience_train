import torch
import numpy as np
import onnx
import onnxruntime as ort

import config as cfg
from model import MiniVGGNet


def export():
    checkpoint = torch.load(str(cfg.BEST_MODEL_PATH), map_location="cpu")
    model = MiniVGGNet(cfg.NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(cfg.ONNX_PATH),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load(str(cfg.ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(cfg.ONNX_PATH), providers=["CPUExecutionProvider"])
    dummy_np = dummy.numpy()
    ort_out = session.run(["logits"], {"input": dummy_np})[0]

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    max_diff = float(np.abs(ort_out - torch_out).max())
    print(f"Exportacao concluida: {cfg.ONNX_PATH}")
    print(f"Max diff ONNX vs PyTorch: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("Validacao PASSOU (diff < 1e-5)")
    else:
        print(f"AVISO: diff={max_diff:.2e} esta acima do limiar 1e-5")
