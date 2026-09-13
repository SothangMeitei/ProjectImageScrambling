import numpy as np
import matplotlib.pyplot as plt

class EntropyAnalyzer:
    """Handles Information Entropy Analysis and Distribution."""
    
    @staticmethod
    def calculate_global(image: np.ndarray) -> float:
        flat = image.flatten()
        counts = np.bincount(flat, minlength=256)
        probabilities = counts[counts > 0] / float(len(flat))
        return -np.sum(probabilities * np.log2(probabilities))

    @staticmethod
    def calculate_local(image: np.ndarray, block_size: int = 64) -> float:
        h, w = image.shape[:2]
        if h < block_size or w < block_size:
            return 0.0
            
        entropies = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = image[y:y+block_size, x:x+block_size]
                entropies.append(EntropyAnalyzer.calculate_global(block))
        return float(np.mean(entropies))

    @staticmethod
    def plot_histograms(plain: np.ndarray, cipher: np.ndarray, output_path: str):
        """Plots the pixel intensity distribution of plain vs cipher."""
        import os
        output_path = os.path.abspath(output_path)
        base_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        if base_name.endswith(".png.png"):
            output_path = os.path.join(base_dir, base_name[:-4])
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(plain.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
        ax1.set_title("Plaintext Histogram (Natural Distribution)")
        ax1.set_xlim([0, 256])
        
        ax2.hist(cipher.flatten(), bins=256, range=(0, 256), color='red', alpha=0.7)
        ax2.set_title("Ciphertext Histogram (Chaotic Uniformity)")
        ax2.set_xlim([0, 256])
        
        plt.tight_layout()
        try:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            plt.savefig(output_path)
        except Exception as e:
            print(f"[WARNING]: Could not save histogram {os.path.basename(output_path)}: {e}")
        finally:
            plt.close(fig)