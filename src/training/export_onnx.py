import torch
from pathlib import Path

from src.training.model import build_model


def export(
    checkpoint="artifacts/best_model.pt",
    output="artifacts/model.onnx",
    num_classes=10,
):
    # Build the same architecture, then load the trained weights into it.
    # pretrained=False because we are about to overwrite every weight anyway.
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))

    # Export always runs in eval mode on CPU: the server has no GPU.
    model.eval()

    # ONNX records the graph by tracing one example input through the model.
    dummy = torch.randn(1, 3, 64, 64)

    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["logits"],
        # Batch size is marked dynamic so the served model accepts any number
        # of images, not just the single image used for tracing.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    print(f"Exported to {output}")


if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    export()