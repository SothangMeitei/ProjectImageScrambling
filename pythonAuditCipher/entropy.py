import numpy as np

class EntropyAnalyzer:
    """Handles Information Entropy Analysis (Measures randomness)."""

    """this is to measure the global and the local entropy of the image encrypted file"""

    @staticmethod
    def calculate_global(image: np.ndarray) -> float:
        flat = image.flatten()
        counts = np.bincount(flat, minlength=256)
        probabilities = counts[counts > 0] / float(len(flat))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    @staticmethod
    def calculate_local(image: np.ndarray, block_size: int = 8) -> float:
        """Calculates Non-Overlapping Local Shannon Entropy."""
        h, w = image.shape[:2]
        entropies = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = image[y:y+block_size, x:x+block_size]
                entropies.append(EntropyAnalyzer.calculate_global(block))
        return float(np.mean(entropies))