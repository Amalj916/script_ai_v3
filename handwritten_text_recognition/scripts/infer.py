import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from train_cnn import CNN


class OCRPredictor:
    def __init__(self, model_path, num_classes=62):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = CNN(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def predict(self, image_path):
        """
        Predict character class from image
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("L")
            image = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)
                return predicted.item()

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise e


def main():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    model_path = os.path.join(project_dir, "models", "cnn_model.pth")
    test_dir = os.path.join(project_dir, "dataset", "test")

    try:
        # Initialize predictor
        predictor = OCRPredictor(model_path)
        print("✅ Model loaded successfully!")

        # Test on a few images
        for class_folder in sorted(os.listdir(test_dir))[:5]:  # Test first 5 classes
            class_path = os.path.join(test_dir, class_folder)
            if os.path.isdir(class_path):
                print(f"\nTesting class {class_folder}:")

                for img_name in os.listdir(class_path)[:3]:  # Test first 3 images
                    img_path = os.path.join(class_path, img_name)
                    pred = predictor.predict(img_path)
                    print(f"Image: {img_name}, Predicted: {pred}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
