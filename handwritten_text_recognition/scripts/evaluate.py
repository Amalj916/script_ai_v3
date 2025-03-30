import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from handwritten_text_recognition.scripts.dataloader import CharacterDataset
from train_cnn import CNN
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    model_path = os.path.join(project_dir, "models", "cnn_model.pth")
    test_dir = os.path.join(project_dir, "dataset", "test")
    output_dir = os.path.join(project_dir, "outputs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    try:
        # Load test dataset
        test_dataset = CharacterDataset(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Load model
        model = CNN(num_classes=len(test_dataset.classes)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Evaluate
        predictions, labels = evaluate_model(model, test_loader, device)

        # Generate classification report
        report = classification_report(
            labels, predictions, target_names=test_dataset.classes, digits=4
        )

        print("\nClassification Report:")
        print(report)

        # Save report to file
        with open(os.path.join(output_dir, "evaluation_report.txt"), "w") as f:
            f.write(report)

        # Plot confusion matrix
        plot_confusion_matrix(
            labels,
            predictions,
            classes=test_dataset.classes,
            output_path=os.path.join(output_dir, "confusion_matrix.png"),
        )

        print("✅ Evaluation completed!")
        print(f"Results saved in: {output_dir}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")


if __name__ == "__main__":
    main()
