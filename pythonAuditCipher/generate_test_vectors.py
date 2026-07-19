import numpy as np
import cv2
import os
import argparse

def generate_uniform_image(filename: str, width: int, height: int, r: int, g: int, b: int, a: int = None):
    """
    Generates a pure, zero-entropy uniform color image.
    
    Args:
        filename (str): The output path (e.g., 'assets/pure_black.png').
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        r, g, b (int): Color channel values (0-255).
        a (int, optional): Alpha channel value. If None, generates a 3-channel RGB image.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Note: OpenCV's internal memory layout is strictly BGR (Blue, Green, Red).
    # We must format the tuple inversely to match the layout before writing to disk.
    if a is not None:
        # 4-Channel (BGRA) - 32 bits per pixel
        image_data = np.full((height, width, 4), (b, g, r, a), dtype=np.uint8)
        channels = "RGBA"
    else:
        # 3-Channel (BGR) - 24 bits per pixel (Matches your C++ stbi_load layout)
        image_data = np.full((height, width, 3), (b, g, r), dtype=np.uint8)
        channels = "RGB"

    # Save the raw array to disk as a lossless PNG
    cv2.imwrite(filename, image_data)
    
    # Calculate uncompressed size in Megabytes
    size_mb = (width * height * (4 if a is not None else 3)) / (1024 * 1024)
    
    print(f"[ASSET GENERATED]: '{filename}'")
    print(f"  -> Resolution : {width}x{height}")
    print(f"  -> Channels   : {channels}")
    print(f"  -> RGB Values : ({r}, {g}, {b})")
    print(f"  -> Raw Size   : {size_mb:.2f} MB\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Uniform Zero-Entropy Images.")
    parser.add_argument("filename", type=str, help="Output file path (e.g., out.png)")
    parser.add_argument("width", type=int, help="Image width")
    parser.add_argument("height", type=int, help="Image height")
    parser.add_argument("r", type=int, help="Red value (0-255)")
    parser.add_argument("g", type=int, help="Green value (0-255)")
    parser.add_argument("b", type=int, help="Blue value (0-255)")
    parser.add_argument("-a", "--alpha", type=int, default=None, help="Alpha value (0-255)")

    args = parser.parse_args()
    generate_uniform_image(args.filename, args.width, args.height, args.r, args.g, args.b, args.alpha)