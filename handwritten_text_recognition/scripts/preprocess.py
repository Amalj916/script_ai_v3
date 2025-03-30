import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def preprocess_image(image_path, output_size=(32, 32)):
    """
    Preprocess single image:
    1. Read image
    2. Convert to grayscale
    3. Apply adaptive thresholding
    4. Resize to standard size
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Resize image
        resized = cv2.resize(binary, output_size, interpolation=cv2.INTER_AREA)

        return resized

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def process_dataset(input_dir, output_dir, output_size=(32, 32)):
    """
    Process entire dataset maintaining folder structure
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each class folder
    for class_name in os.listdir(input_dir):
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(input_class_dir):
            continue

        # Create class directory in output
        os.makedirs(output_class_dir, exist_ok=True)

        # Process all images in class directory
        image_files = [
            f
            for f in os.listdir(input_class_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"\nProcessing class {class_name} ({len(image_files)} images)")
        for img_name in tqdm(image_files):
            input_path = os.path.join(input_class_dir, img_name)
            output_path = os.path.join(output_class_dir, img_name)

            # Process image
            processed = preprocess_image(input_path, output_size)
            if processed is not None:
                cv2.imwrite(output_path, processed)


def main():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Define paths for train and test sets
    train_input = os.path.join(project_dir, "dataset", "train")
    test_input = os.path.join(project_dir, "dataset", "test")

    train_output = os.path.join(project_dir, "dataset", "processed", "train")
    test_output = os.path.join(project_dir, "dataset", "processed", "test")

    try:
        # Process training set
        print("Processing training set...")
        process_dataset(train_input, train_output)

        # Process test set
        print("\nProcessing test set...")
        process_dataset(test_input, test_output)

        print("\n✅ Preprocessing completed!")
        print(f"Processed images saved to: {os.path.dirname(train_output)}")

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")


if __name__ == "__main__":
    main()
