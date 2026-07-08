import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class RobustnessAnalyzer:
    """Handles structural integrity, data loss staging, and noise attacks."""
    
    @staticmethod
    def stage_cropping_attack(cipher: np.ndarray, output_path: str, fraction: float = 0.125):
        attacked = cipher.copy()
        h, w = attacked.shape[:2]
        block_w, block_h = int(w * np.sqrt(fraction)), int(h * np.sqrt(fraction))
        sy, sx = (h - block_h) // 2, (w - block_w) // 2
        
        attacked[sy:sy+block_h, sx:sx+block_w] = 0
        cv2.imwrite(output_path, attacked)

    @staticmethod
    def stage_noise_attack(cipher: np.ndarray, output_path: str, density: float = 0.05):
        attacked = cipher.copy()
        random_mat = np.random.rand(*attacked.shape[:2])
        
        # Apply salt and pepper across all channels if image is 3D
        mask_pepper = random_mat < (density / 2)
        mask_salt = (random_mat >= (density / 2)) & (random_mat < density)
        
        if len(attacked.shape) == 3:
            attacked[mask_pepper] = [0, 0, 0]
            attacked[mask_salt] = [255, 255, 255]
        else:
            attacked[mask_pepper] = 0
            attacked[mask_salt] = 255
            
        cv2.imwrite(output_path, attacked)

    @staticmethod
    def evaluate_quality(original: np.ndarray, recovered: np.ndarray) -> tuple:
        """Returns (PSNR, SSIM) between two arrays."""
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            recovered = cv2.cvtColor(recovered, cv2.COLOR_BGR2GRAY)
            
        psnr_val = psnr(original, recovered)
        ssim_val, _ = ssim(original, recovered, full=True)
        return psnr_val, ssim_val