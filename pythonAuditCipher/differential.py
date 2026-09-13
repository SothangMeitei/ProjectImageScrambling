import numpy as np

class DifferentialAnalyzer:
    """Handles Differential Cryptanalysis metrics (Avalanche effect)."""
    
    @staticmethod
    def calculate_npcr(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
        """Number of Pixels Change Rate."""
        if cipher1.shape != cipher2.shape:
            raise ValueError(f"Shape mismatch: {cipher1.shape} vs {cipher2.shape}")
        c1_flat, c2_flat = cipher1.flatten(), cipher2.flatten()
        diff_count = np.sum(c1_flat != c2_flat)
        return (diff_count / len(c1_flat)) * 100.0

    @staticmethod
    def calculate_uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
        """Unified Average Changing Intensity."""
        if cipher1.shape != cipher2.shape:
            raise ValueError(f"Shape mismatch: {cipher1.shape} vs {cipher2.shape}")
        c1_flat = cipher1.flatten().astype(np.float64)
        c2_flat = cipher2.flatten().astype(np.float64)
        diff_sum = np.sum(np.abs(c1_flat - c2_flat))
        return (diff_sum / (255.0 * len(c1_flat))) * 100.0