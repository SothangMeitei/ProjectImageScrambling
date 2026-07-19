import cv2
import os
from .entropy import EntropyAnalyzer
from .differential import DifferentialAnalyzer
from .correlation import CorrelationAnalyzer
from .robustness import RobustnessAnalyzer

class CipherAuditSuite:
    """The central Controller that composes all test modules."""
    
    def __init__(self, plain_path: str, cipher_path: str):
        self.plain = self._load(plain_path)
        self.cipher = self._load(cipher_path)
        self.plain_path = plain_path
        self.cipher_path = cipher_path

    def _load(self, path: str):
        img = cv2.imread(path)
        if img is None: raise ValueError(f"Failed to load image at {path}")
        return img

    def run_entropy_audit(self):
        g_ent = EntropyAnalyzer.calculate_global(self.cipher)
        l_ent = EntropyAnalyzer.calculate_local(self.cipher)
        print(f"Global Entropy : {g_ent:.5f} / 8.0")
        print(f"Local Entropy  : {l_ent:.5f}")

    def run_differential_audit(self, diff_cipher_path: str):
        diff_cipher = self._load(diff_cipher_path)
        npcr = DifferentialAnalyzer.calculate_npcr(self.cipher, diff_cipher)
        uaci = DifferentialAnalyzer.calculate_uaci(self.cipher, diff_cipher)
        print(f"NPCR : {npcr:.4f}%")
        print(f"UACI : {uaci:.4f}%")

    def run_correlation_audit(self, plot_out: str = "outputs/correlation"):
        for d in ["horizontal", "vertical", "diagonal"]:
            p_corr = CorrelationAnalyzer.calculate(self.plain, d)
            c_corr = CorrelationAnalyzer.calculate(self.cipher, d)
            print(f"{d.capitalize():<12} | Plain: {p_corr:8.5f} | Cipher: {c_corr:8.5f}")
        
        CorrelationAnalyzer.plot_scatter(self.plain, self.cipher, plot_out)
        print(f"[+] Scatter plot saved to {plot_out}")

    def stage_attacks(self, crop_out: str, noise_out: str):
        RobustnessAnalyzer.stage_cropping_attack(self.cipher, crop_out)
        RobustnessAnalyzer.stage_noise_attack(self.cipher, noise_out)
        print("[+] Attacks staged. Feed these to the C++ decryption engine.")

    def verify_recovery(self, recovered_crop_path: str, recovered_noise_path: str):
        rec_crop = self._load(recovered_crop_path)
        rec_noise = self._load(recovered_noise_path)
        
        psnr_c, ssim_c = RobustnessAnalyzer.evaluate_quality(self.plain, rec_crop)
        psnr_n, ssim_n = RobustnessAnalyzer.evaluate_quality(self.plain, rec_noise)
        
        print(f"Crop Attack  | PSNR: {psnr_c:5.2f} dB | SSIM: {ssim_c:.4f}")
        print(f"Noise Attack | PSNR: {psnr_n:5.2f} dB | SSIM: {ssim_n:.4f}")

    def run_entropy_audit(self, plot_out: str = "outputs/histogram.png"):
        g_ent = EntropyAnalyzer.calculate_global(self.cipher)
        l_ent = EntropyAnalyzer.calculate_local(self.cipher)
        print(f"Global Entropy : {g_ent:.5f} / 8.0")
        print(f"Local Entropy  : {l_ent:.5f}")
        
        # NEW: Output the histogram
        EntropyAnalyzer.plot_histograms(self.plain, self.cipher, plot_out)
        print(f"[+] Histogram saved to {plot_out}")

    def run_key_sensitivity_audit(plain_path: str, cipher_path_key1: str, cipher_path_key2: str):
        c1 = cv2.imread(cipher_path_key1)
        c2 = cv2.imread(cipher_path_key2)
        
        from pythonAuditCipher.differential import DifferentialAnalyzer
        npcr = DifferentialAnalyzer.calculate_npcr(c1, c2)
        uaci = DifferentialAnalyzer.calculate_uaci(c1, c2)
        
        print("\n=== KEY SENSITIVITY TEST ===")
        print(f"NPCR (Should be > 99.6%) : {npcr:.4f}%")
        print(f"UACI (Should be ~ 33.4%) : {uaci:.4f}%")
    
    def run_nist_image_audit(self):
        """Dynamically binds the NIST analysis suite into the runtime image loop."""
        from .NIST_key_stream_test import NISTAnalyzer
        NISTAnalyzer.run_suite(self.cipher)