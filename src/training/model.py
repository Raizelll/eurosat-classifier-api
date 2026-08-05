import torch.nn as nn
from torchvision import models


def build_model(num_classes=10, pretrained=True):
    # Load ResNet-18. With pretrained=True we get weights already trained on ImageNet.
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # The original final layer outputs 1000 values, one per ImageNet class.
    # We replace it with a new layer of 10 outputs, one per EuroSAT class.
    # in_features is read from the old layer so we never hardcode the number.
    in_features = model.fc.in_features  # fc : full connected layer
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    import torch

    model = build_model()
    # Feed one fake batch to confirm the shapes line up before real training.
    dummy = torch.randn(2, 3, 64, 64)
    output = model(dummy)
    print("Output shape:", output.shape)