import numpy as np
import cv2
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

@dataclass
class CropBox:
    x: int; y: int; w: int; h: int

class CropConfiguration:
    def __init__(self, mode: str = "random", 
                 custom_boxes: Optional[List[CropBox]] = None,
                 num_boxes: int = 3, 
                 size_range: Tuple[int, int] = (50, 200)):
        """
        mode: "custom" (uses custom_boxes) or "random" (generates based on params)
        """
        self.mode = mode
        self.custom_boxes = custom_boxes or []
        self.num_boxes = num_boxes
        self.size_range = size_range

    def generate_boxes(self, img_w: int, img_h: int) -> List[CropBox]:
        if self.mode == "custom":
            return self.custom_boxes
            
        boxes = []
        for _ in range(self.num_boxes):
            w = random.randint(self.size_range[0], self.size_range[1])
            h = random.randint(self.size_range[0], self.size_range[1])
            x = random.randint(0, max(0, img_w - w))
            y = random.randint(0, max(0, img_h - h))
            boxes.append(CropBox(x, y, w, h))
        return boxes


class RobustnessAnalyzer:
    @staticmethod
    def stage_advanced_cropping(cipher: np.ndarray, config: CropConfiguration, output_path: str):
        attacked = cipher.copy()
        h, w = attacked.shape[:2]
        
        # Generator processes the config and spits out bounding boxes
        boxes = config.generate_boxes(w, h)
        
        # Apply the destruction
        for b in boxes:
            # Ensure boundaries are safe
            y_end = min(b.y + b.h, h)
            x_end = min(b.x + b.w, w)
            attacked[b.y:y_end, b.x:x_end] = 0
            
        cv2.imwrite(output_path, attacked)

    @staticmethod
    def stage_noise_attack(cipher: np.ndarray, output_path: str, density: float = 0.05):
        attacked = cipher.copy()
        random_mat = np.random.rand(*attacked.shape[:2])
        mask_pepper = random_mat < (density / 2)
        mask_salt = (random_mat >= (density / 2)) & (random_mat < density)
        if len(attacked.shape) == 3:
            attacked[mask_pepper] = [0, 0, 0]
            attacked[mask_salt] = [255, 255, 255]
        else:
            attacked[mask_pepper] = 0; attacked[mask_salt] = 255
        cv2.imwrite(output_path, attacked)

    @staticmethod
    def evaluate_quality(original: np.ndarray, recovered: np.ndarray) -> tuple:
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            recovered = cv2.cvtColor(recovered, cv2.COLOR_BGR2GRAY)
        psnr_val = psnr(original, recovered)
        ssim_val, _ = ssim(original, recovered, full=True)
        return psnr_val, ssim_val