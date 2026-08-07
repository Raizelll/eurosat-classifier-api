import numpy as np
import onnxruntime as ort
import torch

from src.training.model import build_model


def verify(checkpoint="artifacts/best_model.pt", onnx_path="artifacts/model.onnx"):
    # Same input for both models, so any difference comes from the export.
    dummy = torch.randn(2, 3, 64, 64)

    model = build_model(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(["logits"], {"input": dummy.numpy()})[0]

    # Tiny numeric differences are expected between backends.
    # 1e-4 is a standard tolerance for this check.
    max_diff = np.abs(torch_out - onnx_out).max()
    print("Max difference:", max_diff)

    assert max_diff < 1e-4, "ONNX output does not match PyTorch"
    print("ONNX export verified")


if __name__ == "__main__":
    verify()