import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from train_cnn import CNN


class CharacterPredictor:
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
            image = Image.open(image_path).convert("L")  # Convert to grayscale

            # Transform image
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                probabilities = torch.softmax(output, dim=1)

                # Get top 3 predictions
                top_prob, top_idx = torch.topk(probabilities, 3)
                top_prob = top_prob[0].cpu().numpy()
                top_idx = top_idx[0].cpu().numpy()

                # Convert to characters
                predictions = []
                for idx, prob in zip(top_idx, top_prob):
                    char = self.idx_to_char[idx]
                    predictions.append((char, prob))

                return predictions

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None


def main():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Model path
    model_path = os.path.join(project_dir, "models", "best_model.pth")

    # Initialize predictor
    predictor = CharacterPredictor(model_path)

    while True:
        # Get input image path
        print("\nEnter the path to your character image (or 'q' to quit):")
        image_path = input().strip('"')  # Remove quotes if present

        if image_path.lower() == "q":
            break

        if os.path.exists(image_path):
            # Get prediction
            predictions = predictor.predict(image_path)

            if predictions:
                print("\nPredictions:")
                print("-" * 30)
                for i, (char, prob) in enumerate(predictions, 1):
                    print(f"{i}. Character: {char} (Confidence: {prob:.2%})")
                print("-" * 30)
        else:
            print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()
