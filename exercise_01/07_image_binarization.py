import os
import cv2

def create_binary_image(image_path):
    # 1. Load the input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from path: {image_path}")
        return

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. First step: Apply a filter (Gaussian Blur)
    # Used for smoothing and noise reduction to prevent false edges
    filtered = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Second step: Apply a differential operator (Sobel)
    # Find gradients along horizontal (X) and vertical (Y) axes
    sobel_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude (overall gradient strength) from both directions
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize and convert back to 8-bit format (0 - 255)
    sobel_normalized = cv2.convertScaleAbs(sobel_magnitude)

    # 4. Third step: Thresholding operation
    # All values above the threshold (e.g., 75) become white (255), others black (0)
    threshold_value = 75
    _, binary_image = cv2.threshold(
        sobel_normalized, threshold_value, 255, cv2.THRESH_BINARY
    )

    # --- Visualization of steps ---
    cv2.imshow("1. Original Grayscale", gray)
    cv2.imshow("2. Filtered (Gaussian Blur)", filtered)
    cv2.imshow("3. Differential Operator (Sobel)", sobel_normalized)
    cv2.imshow("4. Binary Image (Threshold)", binary_image)

    # Create the 'output' directory if it doesn't already exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save the final result inside the output folder
    output_path = os.path.join(output_dir, "binary_output.jpg")
    cv2.imwrite(output_path, binary_image)
    print(f"Done! Binary image saved as '{output_path}'.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with the path to your image
    image_path = "cats.jpg"
    create_binary_image(image_path)
