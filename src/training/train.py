import torch
import torch.nn as nn
from pathlib import Path

from src.training.data import get_dataloaders
from src.training.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Clear gradients left over from the previous batch.
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Compute gradients, then update the weights.
        loss.backward()
        optimizer.step()

        #the loss value returned is already averaged over the batch, 
        # and we want to add the total contribution of this batch to the running loss, so we multiply by the batch size.
        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    # Evaluation mode: no dropout, BatchNorm uses stored statistics.
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # No gradients needed here, which saves memory and time.
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main(epochs=5, batch_size=64, lr=1e-3, out_dir="artifacts"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, classes = get_dataloaders( batch_size=batch_size )

    model = build_model(num_classes=len(classes)).to(device)

    # Standard loss for multi-class classification. It expects raw logits.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Path(out_dir).mkdir(exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch( model, train_loader, criterion, optimizer, device )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Keep only the checkpoint with the best validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(out_dir) / "best_model.pt")
            print(f"  saved new best model (val acc {val_acc:.4f})")

    # The test set is touched once, at the very end, with the best checkpoint.
    model.load_state_dict( torch.load(Path(out_dir) / "best_model.pt", weights_only=True) )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")


if __name__ == "__main__":
    main()

