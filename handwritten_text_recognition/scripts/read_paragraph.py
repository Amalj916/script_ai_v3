import cv2
import numpy as np
import torch
from PIL import Image
import os
from torchvision import transforms
from train_cnn import CNN


class ParagraphReader:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the trained model
        self.model = CNN(num_classes=36)  # 36 classes as in your training
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Define transforms (same as training)
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

    def preprocess_paragraph(self, image):
        """Preprocess paragraph image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    def segment_lines(self, binary_image):
        """Segment image into lines"""
        # Get horizontal projection
        h_proj = np.sum(binary_image, axis=1)

        # Find line boundaries
        lines = []
        start = None
        min_line_height = 20

        for i, proj in enumerate(h_proj):
            if proj > 0 and start is None:
                start = i
            elif (proj == 0 or i == len(h_proj) - 1) and start is not None:
                if i - start > min_line_height:
                    lines.append((start, i))
                start = None

        return lines

    def segment_characters(self, line_image):
        """Segment line into characters"""
        # Find contours
        contours, _ = cv2.findContours(
            line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get character regions
        chars = []
        min_width = 5

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > min_width:  # Filter out noise
                chars.append((x, y, w, h))

        # Sort left to right
        chars.sort(key=lambda x: x[0])

        return chars

    def recognize_character(self, char_image):
        """Recognize a single character"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(char_image)

            # Apply transforms
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                pred_idx = output.argmax().item()
                confidence = torch.softmax(output, dim=1).max().item()

                return self.idx_to_char.get(pred_idx, ""), confidence

        except Exception as e:
            print(f"Error recognizing character: {str(e)}")
            return "", 0.0

    def read_paragraph(self, image_path, confidence_threshold=0.5):
        """Convert paragraph image to text"""
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            binary = self.preprocess_paragraph(image)

            # Get lines
            line_regions = self.segment_lines(binary)

            # Process each line
            paragraph_text = []

            for line_start, line_end in line_regions:
                line_image = binary[line_start:line_end, :]
                char_regions = self.segment_characters(line_image)

                line_text = []
                prev_x = 0

                # Process each character
                for x, y, w, h in char_regions:
                    # Add space if there's a large gap
                    if x - prev_x > w * 1.5:
                        line_text.append(" ")

                    # Extract and recognize character
                    char_img = line_image[y : y + h, x : x + w]
                    char, conf = self.recognize_character(char_img)

                    # Add character if confidence is above threshold
                    if conf > confidence_threshold:
                        line_text.append(char)

                    prev_x = x + w

                # Add line to paragraph
                if line_text:
                    paragraph_text.append("".join(line_text))

            return "\n".join(paragraph_text)

        except Exception as e:
            print(f"Error processing paragraph: {str(e)}")
            return ""


def main():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Initialize reader with the best model
    model_path = os.path.join(project_dir, "models", "best_model.pth")
    reader = ParagraphReader(model_path)

    # Create output directory
    output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Test on a paragraph image
    print("\nEnter the path to your paragraph image:")
    image_path = input().strip('"')  # Remove quotes if present

    if os.path.exists(image_path):
        # Read paragraph
        print("\nProcessing paragraph...")
        text = reader.read_paragraph(image_path)

        print("\nRecognized Text:")
        print("-" * 50)
        print(text)
        print("-" * 50)

        # Save result
        output_file = os.path.join(output_dir, "recognized_text.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"\nText saved to: {output_file}")
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()
