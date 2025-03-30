import cv2
import numpy as np
import torch
from PIL import Image
import os
from torchvision import transforms
from train_cnn import CNN


class SimpleReader:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the trained model
        self.model = CNN(num_classes=36)  # 36 classes as in your training
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define image transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Character mapping (0-9, A-Z)
        self.idx_to_char = {}
        # Add digits (0-9)
        for i in range(10):
            self.idx_to_char[i] = str(i)
        # Add uppercase letters (A-Z)
        for i in range(26):
            self.idx_to_char[i + 10] = chr(65 + i)

    def preprocess_image(self, image_path):
        """Basic preprocessing for a single image"""
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Apply transforms
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        return tensor

    def predict(self, image_path):
        """Predict character from image"""
        try:
            # Preprocess image
            tensor = self.preprocess_image(image_path)

            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                pred_idx = output.argmax().item()
                predicted_char = self.idx_to_char.get(pred_idx, "")

                return predicted_char, pred_idx

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None


def main():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Model path
    model_path = os.path.join(project_dir, "models", "cnn_model.pth")

    # Initialize reader
    reader = SimpleReader(model_path)

    # Test directory (path to your test images)
    test_dir = os.path.join(project_dir, "dataset", "test")

    print("\nTesting character recognition...")
    print("-" * 50)

    # Test a few images from different classes
    for class_name in sorted(os.listdir(test_dir))[:5]:  # Test first 5 classes
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"\nTesting class {class_name}:")

            # Test first 3 images from this class
            for img_name in sorted(os.listdir(class_dir))[:3]:
                if img_name.endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, img_name)
                    pred_char, pred_idx = reader.predict(img_path)

                    print(f"Image: {img_name}")
                    print(f"Predicted: {pred_char} (index: {pred_idx})")


if __name__ == "__main__":
    main()
