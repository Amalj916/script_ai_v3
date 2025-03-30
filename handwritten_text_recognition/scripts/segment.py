import cv2
import numpy as np
import os


def segment_characters(image_path):
    """
    Segment characters from an image:
    1. Preprocess image
    2. Find contours
    3. Extract character regions
    """
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left-to-right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    characters = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out noise
        if w * h > 100:  # Minimum area threshold
            char_region = gray[y : y + h, x : x + w]
            characters.append({"image": char_region, "bbox": (x, y, w, h)})

    return characters


def save_segments(image_path, output_dir):
    """Save segmented characters as individual images"""
    try:
        characters = segment_characters(image_path)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save each character
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for idx, char in enumerate(characters):
            output_path = os.path.join(output_dir, f"{base_name}_char_{idx}.png")
            cv2.imwrite(output_path, char["image"])

        return len(characters)

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return 0


if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    test_image = os.path.join(project_dir, "dataset", "test", "images", "test1.png")
    output_dir = os.path.join(project_dir, "dataset", "test", "segments")

    num_chars = save_segments(test_image, output_dir)
    print(f"✅ Segmented {num_chars} characters!")
