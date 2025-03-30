import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from train_cnn import CNN
from tqdm import tqdm


class CharacterTester:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the trained model
        self.model = CNN(num_classes=36)
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Character mapping
        self.idx_to_char = {}
        # Add digits (0-9)
        for i in range(10):
            self.idx_to_char[i] = str(i)
        # Add uppercase letters (A-Z)
        for i in range(26):
            self.idx_to_char[i + 10] = chr(65 + i)

    def predict(self, image_path):
        """Predict character from image"""
        try:
            # Read image
            image = Image.open(image_path).convert("L")

            # Transform image
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                probabilities = torch.softmax(output, dim=1)
                pred_idx = output.argmax().item()
                confidence = probabilities[0][pred_idx].item()

                return self.idx_to_char[pred_idx], confidence, pred_idx

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, 0.0, None


def test_model():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Model path
    model_path = os.path.join(project_dir, "models", "best_model.pth")

    # Initialize tester
    tester = CharacterTester(model_path)

    # Test directory
    test_dir = os.path.join(project_dir, "dataset", "test")

    print("\nTesting character recognition...")
    print("-" * 50)

    total_correct = 0
    total_samples = 0
    class_accuracies = {}

    # Test all classes
    for class_name in sorted(os.listdir(test_dir)):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\nTesting class {class_name}:")
        correct = 0
        samples = 0

        # Test all images in this class
        for img_name in sorted(os.listdir(class_dir)):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(class_dir, img_name)
            pred_char, confidence, pred_idx = tester.predict(img_path)

            # Update statistics
            samples += 1
            if pred_char == class_name:
                correct += 1

            print(f"Image: {img_name}")
            print(f"True: {class_name}, Predicted: {pred_char}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 30)

        # Calculate class accuracy
        class_acc = (correct / samples) * 100 if samples > 0 else 0
        class_accuracies[class_name] = class_acc
        print(f"Class {class_name} Accuracy: {class_acc:.2f}% ({correct}/{samples})")

        total_correct += correct
        total_samples += samples

    # Print overall results
    print("\nOverall Results:")
    print("-" * 50)
    print(
        f"Total Accuracy: {(total_correct/total_samples)*100:.2f}% ({total_correct}/{total_samples})"
    )

    # Print class-wise accuracies sorted by accuracy
    print("\nClass-wise Accuracies (sorted):")
    print("-" * 50)
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    for class_name, acc in sorted_classes:
        print(f"{class_name}: {acc:.2f}%")


if __name__ == "__main__":
    test_model()
