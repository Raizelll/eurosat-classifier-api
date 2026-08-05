import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Mean and std of the RGB channels computed over the whole ImageNet dataset.
# We use ImageNet values (not EuroSAT ones) because ResNet-18 was pretrained on
# ImageNet: its weights expect inputs with this exact distribution.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transform():
    # An operations applied to every image the moment it is loaded.
    return transforms.Compose([
        # Force a fixed size. EuroSAT images are already 64x64, but we set it-
        # -explicitly because the API will later receive images of any size.
        transforms.Resize((64, 64)),

        # Convert PIL image to a tensor and scale pixel values from 0-255 to 0-1.
        transforms.ToTensor(),

        # normalized = (value - mean) / std
        # Centers values around 0, which makes training faster and more stable.
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_dataloaders(root="data", batch_size=64, seed=42):
    # Load EuroSAT and attach the transform recipe to it.
    dataset = datasets.EuroSAT(
        root=root,
        download=True,
        transform=build_transform(),
    )

    # Split sizes: 70% train, 15% validation, 15% test.
    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    # Fixed random seed so the split is identical on every run.
    # Without it, two runs are not comparable. Same principle as the thesis.
    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split( dataset, [n_train, n_val, n_test], generator=generator )

    # shuffle=True only for training: it stops the model from memorizing order.
    # shuffle=False for evaluation: no benefit, and we want a stable order.
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset.classes


# Runs only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Pull one batch to check the shape before training starts.
    # Shape mismatches are the most common training bug: catching them here
    # takes two seconds instead of failing mid-run.
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)
    print("Labels shape:", labels.shape)