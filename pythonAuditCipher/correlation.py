import numpy as np
import matplotlib.pyplot as plt
import os

class CorrelationAnalyzer:
    """Handles Spatial Correlation Analysis and Data Visualization."""
    
    @staticmethod
    def calculate(image: np.ndarray, direction: str = "horizontal", samples: int = 5000) -> float:
        import cv2
        # Convert to 2D grayscale if a 3D color image is passed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        h, w = image.shape
        x_coords = np.random.randint(0, w - 1, samples)
        y_coords = np.random.randint(0, h - 1, samples)

        if direction == "horizontal":
            x1, y1, x2, y2 = x_coords, y_coords, x_coords + 1, y_coords
        elif direction == "vertical":
            x1, y1, x2, y2 = x_coords, y_coords, x_coords, y_coords + 1
        elif direction == "diagonal":
            x1, y1, x2, y2 = x_coords, y_coords, x_coords + 1, y_coords + 1
        else:
            raise ValueError("Direction must be horizontal, vertical, or diagonal")

        v1 = image[y1, x1].astype(np.float64)
        v2 = image[y2, x2].astype(np.float64)

        mean_v1, mean_v2 = np.mean(v1), np.mean(v2)
        numerator = np.sum((v1 - mean_v1) * (v2 - mean_v2))
        denominator = np.sqrt(np.sum((v1 - mean_v1)**2) * np.sum((v2 - mean_v2)**2))
        
        return numerator / denominator if denominator != 0 else 0.0

    @staticmethod
    def plot_scatter(plain: np.ndarray, cipher: np.ndarray, output_path: str, samples: int = 2000):
        import cv2
        if len(plain.shape) == 3: plain = cv2.cvtColor(plain, cv2.COLOR_BGR2GRAY)
        if len(cipher.shape) == 3: cipher = cv2.cvtColor(cipher, cv2.COLOR_BGR2GRAY)

        h, w = plain.shape
        x, y = np.random.randint(0, w - 1, samples), np.random.randint(0, h - 1, samples)

        output_path = os.path.abspath(output_path)
        base_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        # Strip duplicate .png if present
        if base_name.endswith(".png.png"):
            base_name = base_name[:-4]
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

        def _safe_save(fig, target_file):
            try:
                if os.path.exists(target_file):
                    try:
                        os.remove(target_file)
                    except OSError:
                        pass
                fig.savefig(target_file)
            except Exception as e:
                print(f"[WARNING]: Could not save plot {os.path.basename(target_file)}: {e}")
            finally:
                plt.close(fig)

        # --- Horizontal ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.scatter(plain[y, x], plain[y, x + 1], s=1, c='blue', alpha=0.5)
        ax1.set_title("Plaintext: Horizontal")
        ax2.scatter(cipher[y, x], cipher[y, x + 1], s=1, c='red', alpha=0.5)
        ax2.set_title("Ciphertext: Horizontal")
        plt.tight_layout()
        _safe_save(fig, os.path.join(base_dir, f"corr_horizontal_{base_name}"))

        # --- Vertical ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.scatter(plain[y, x], plain[y + 1, x], s=1, c='blue', alpha=0.5)
        ax1.set_title("Plaintext: Vertical")
        ax2.scatter(cipher[y, x], cipher[y + 1, x], s=1, c='red', alpha=0.5)
        ax2.set_title("Ciphertext: Vertical")
        plt.tight_layout()
        _safe_save(fig, os.path.join(base_dir, f"corr_vertical_{base_name}"))

        # --- Diagonal ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.scatter(plain[y, x], plain[y + 1, x + 1], s=1, c='blue', alpha=0.5)
        ax1.set_title("Plaintext: Diagonal")
        ax2.scatter(cipher[y, x], cipher[y + 1, x + 1], s=1, c='red', alpha=0.5)
        ax2.set_title("Ciphertext: Diagonal")
        plt.tight_layout()
        _safe_save(fig, os.path.join(base_dir, f"corr_diagonal_{base_name}"))