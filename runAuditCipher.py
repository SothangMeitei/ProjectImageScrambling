import os
import sys
import glob
import random
import shutil
import subprocess
import argparse
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pythonAuditCipher.orchestrator import CipherAuditSuite
from pythonAuditCipher.robustness import CropConfiguration, RobustnessAnalyzer
from pythonAuditCipher.differential import DifferentialAnalyzer

# ==============================================================================
# AUDIT CONFIGURATION & DIRECTORY HIERARCHY
# ==============================================================================
BASE = "assets"
DIRS = {
    "plain": f"{BASE}/plainText",
    "cipher": f"{BASE}/cipherText",
    "decrypted": f"{BASE}/decryptedCipherText",
    "zero_entropy": f"{BASE}/test_zero_entropy",
    "diff_src": f"{BASE}/test_differential/diff_src",
    "diff_cipher": f"{BASE}/test_differential/diff_cipher",
    "diff_src_1bit": f"{BASE}/test_differential/diff_src_1bit",
    "diff_cipher_1bit": f"{BASE}/test_differential/diff_cipher_1bit",
    "attacks_staged": f"{BASE}/test_robustness/attacks_staged",
    "attacks_recovered": f"{BASE}/test_robustness/attacks_recovered",
    "plots": f"{BASE}/reports/plots",
    "reports": f"{BASE}/reports"
}

TARGET_W, TARGET_H = 1920, 1080


# ==============================================================================
# ENGINE CONTROLLER & BINARY DISCOVERY
# ==============================================================================

def get_actual_filename(dir_path: str, base_name: str) -> str:
    """Helper to support both older engines (which prefix files with encrypted_/decrypted_) and the modern 2D engine."""
    candidates = [
        os.path.join(dir_path, f"decrypted_encrypted_{base_name}"),
        os.path.join(dir_path, f"decrypted_{base_name}"),
        os.path.join(dir_path, f"encrypted_{base_name}"),
        os.path.join(dir_path, base_name)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(dir_path, base_name)

class CipherEngineController:
    """Manages dynamic binary discovery, configuration, and execution of the C++ CUDA engine."""

    CANDIDATE_BINARIES = [
        "DNA_CipherEngine.exe",
        "DNA_CipherEngine_2D.exe",
        "./DNA_CipherEngine.exe",
        "build/DNA_CipherEngine.exe",
        "bin/DNA_CipherEngine.exe"
    ]

    def __init__(self, explicit_binary: Optional[str] = None):
        self.binary_path = self._resolve_binary(explicit_binary)
        print(f"[ENGINE]: Using verified cipher binary -> '{self.binary_path}'")

    @classmethod
    def _resolve_binary(cls, preferred: Optional[str] = None) -> str:
        if preferred and os.path.exists(preferred):
            return preferred
        for cand in cls.CANDIDATE_BINARIES:
            if os.path.exists(cand):
                return cand
        # If none found directly, return default and let run() produce a helpful error
        return "DNA_CipherEngine.exe"

    @staticmethod
    def write_config(mode: str, src_dir: str, out_dir: str, config_path: str = "engine_config.txt"):
        """Writes the key-value config file required by C++ main.cu."""
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"[{mode.upper()}]\n{mode}_src = {src_dir}\n{mode}_out = {out_dir}\n")

    @staticmethod
    def generate_keys(filepath: str = "engine_keys.txt", tweak_chen: bool = False):
        """Generates the chaotic master keys with burn-in parameterization."""
        chen_k1, chen_k2, chen_k3 = 35.0, 3.0, 28.0
        chen_x, chen_y, chen_z = 0.1234567, 0.5432198, 0.9876543
        if tweak_chen:
            chen_x += 0.000001

        lor_a, lor_b, lor_c, lor_r = 10.0, (8.0 / 3.0), 46.0, 2.0
        lor_x, lor_y, lor_z, lor_w = 12.0, 0.7194113, 0.8156727, 0.2946892

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("[CHEN]\n")
            f.write(f"{chen_k1} {chen_k2} {chen_k3} 500000 {chen_x:.15f} {chen_y} {chen_z}\n")
            f.write("[LORENZ]\n")
            f.write(f"{lor_a} {lor_b} {lor_c} {lor_r} {lor_x} 500000 {lor_y} {lor_z} {lor_w} 0.4389124\n")

    def run(self, mode: str):
        """Dispatches execution of the C++ CUDA cipher engine."""
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(
                f"[FATAL]: Cipher binary '{self.binary_path}' not found! "
                "Ensure you compile the CUDA engine (e.g. nvcc / cmake) before running audit."
            )
        subprocess.run([self.binary_path, mode], check=True)


# ==============================================================================
# ASSET MANAGEMENT & VECTOR SYNTHESIS
# ==============================================================================
class AssetManager:
    """Handles folder hierarchy creation, image normalization, and zero-entropy test synthesis."""

    @staticmethod
    def initialize_workspace():
        for path in DIRS.values():
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def generate_zero_entropy_vectors():
        """Synthesizes pure uniform test vectors for edge-case entropy audits."""
        black_path = os.path.join(DIRS["zero_entropy"], "pure_black.png")
        white_path = os.path.join(DIRS["zero_entropy"], "pure_white.png")
        if not os.path.exists(black_path):
            cv2.imwrite(black_path, np.full((TARGET_H, TARGET_W, 3), (0, 0, 0), dtype=np.uint8))
        if not os.path.exists(white_path):
            cv2.imwrite(white_path, np.full((TARGET_H, TARGET_W, 3), (255, 255, 255), dtype=np.uint8))

    @staticmethod
    def standardize_plaintexts() -> List[str]:
        """Normalizes all input plaintext frames to target Full HD dimensions."""
        plain_files = sorted(glob.glob(f"{DIRS['plain']}/*.*"))
        if not plain_files:
            print(f"[WARNING]: No images found in '{DIRS['plain']}'. Please place input frames there.")
            return []

        valid_files = []
        for idx, f in enumerate(plain_files):
            img = cv2.imread(f)
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < TARGET_H or w < TARGET_W:
                print(f"[WARNING]: Skipping {f} - resolution {w}x{h} is below target {TARGET_W}x{TARGET_H}")
                continue
            if h > TARGET_H or w > TARGET_W:
                img = img[0:TARGET_H, 0:TARGET_W]

            target_path = os.path.join(DIRS["plain"], f"frame_{len(valid_files)}.png")
            cv2.imwrite(target_path, img)
            if target_path != f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
            valid_files.append(target_path)

        return valid_files

    @staticmethod
    def synthesize_differential_sources(plain_files: List[str], flips_50: int = 50):
        """Prepares controlled 1-bit and 50-bit plaintext perturbed variants."""
        for d in [DIRS["diff_src"], DIRS["diff_src_1bit"]]:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        for plain_path in plain_files:
            fname = os.path.basename(plain_path)
            img = cv2.imread(plain_path)
            if img is None:
                continue
            h, w, c = img.shape

            # 50-bit random flips
            img_50 = img.copy()
            for _ in range(flips_50):
                img_50[random.randint(0, h-1), random.randint(0, w-1), random.randint(0, c-1)] ^= (1 << random.randint(0, 7))
            cv2.imwrite(os.path.join(DIRS["diff_src"], fname), img_50)

            # 1-bit random flip
            img_1 = img.copy()
            img_1[random.randint(0, h-1), random.randint(0, w-1), random.randint(0, c-1)] ^= (1 << random.randint(0, 7))
            cv2.imwrite(os.path.join(DIRS["diff_src_1bit"], fname), img_1)


# ==============================================================================
# AUDIT STAGES
# ==============================================================================
class AuditPipeline:
    """Executes the comprehensive multi-phase security evaluation."""

    def __init__(self, engine: CipherEngineController):
        self.engine = engine

    def run_key_avalanche_stage(self, plain_files: List[str], report) -> Tuple[List[float], List[float]]:
        """Evaluates avalanche sensitivity with respect to infinitesimal key variation."""
        print("[PIPELINE]: Running Key Sensitivity Test...")
        shutil.rmtree(DIRS["cipher"], ignore_errors=True)
        os.makedirs(DIRS["cipher"], exist_ok=True)
        self.engine.write_config("encrypt", DIRS["plain"], DIRS["cipher"])
        self.engine.generate_keys("engine_keys.txt", tweak_chen=False)
        self.engine.run("encrypt")

        backup_dir = f"{BASE}/cipherText_Base_Backup"
        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.copytree(DIRS["cipher"], backup_dir)

        # Clear cipher folder for mutated run
        shutil.rmtree(DIRS["cipher"], ignore_errors=True)
        os.makedirs(DIRS["cipher"], exist_ok=True)

        try:
            self.engine.generate_keys("engine_keys.txt", tweak_chen=True)
            self.engine.run("encrypt")

            report.write("=" * 60 + "\n")
            report.write("  KEY SENSITIVITY TEST  (Avalanche Effect on Keys)\n")
            report.write("  Key Tweak: Chen x0 += 0.000001\n")
            report.write("  Expected : NPCR > 99.6% | UACI ~ 33.4%\n")
            report.write("=" * 60 + "\n")
            report.write(f"  {'File':<20} {'NPCR':>10} {'UACI':>10}\n")
            report.write("  " + "-" * 44 + "\n")

            npcr_vals, uaci_vals = [], []
            for p in plain_files:
                fname = os.path.basename(p)
                b_img = cv2.imread(get_actual_filename(backup_dir, fname))
                m_img = cv2.imread(get_actual_filename(DIRS["cipher"], fname))
                if b_img is not None and m_img is not None:
                    npcr = DifferentialAnalyzer.calculate_npcr(b_img, m_img)
                    uaci = DifferentialAnalyzer.calculate_uaci(b_img, m_img)
                    npcr_vals.append(npcr)
                    uaci_vals.append(uaci)
                    report.write(f"  {fname:<20} {npcr:>9.4f}% {uaci:>9.4f}%\n")

            if npcr_vals:
                report.write("  " + "-" * 44 + "\n")
                avg_n = sum(npcr_vals) / len(npcr_vals)
                avg_u = sum(uaci_vals) / len(uaci_vals)
                report.write(f"  {'AVERAGE':<20} {avg_n:>9.4f}% {avg_u:>9.4f}%\n")
            report.write("\n")
            return npcr_vals, uaci_vals

        finally:
            shutil.rmtree(DIRS["cipher"], ignore_errors=True)
            os.rename(backup_dir, DIRS["cipher"])
            self.engine.generate_keys("engine_keys.txt", tweak_chen=False)
            print("[PIPELINE]: Baseline cipher directory restored.")

    def run_differential_encryption_stage(self):
        """Encrypts 50-bit and 1-bit perturbed source vectors."""
        print("[PIPELINE]: Running Differential Encryption (50-bit flip)...")
        shutil.rmtree(DIRS["diff_cipher"], ignore_errors=True)
        os.makedirs(DIRS["diff_cipher"], exist_ok=True)
        self.engine.write_config("encrypt", DIRS["diff_src"], DIRS["diff_cipher"])
        self.engine.run("encrypt")

        print("[PIPELINE]: Running Differential Encryption (1-bit flip)...")
        shutil.rmtree(DIRS["diff_cipher_1bit"], ignore_errors=True)
        os.makedirs(DIRS["diff_cipher_1bit"], exist_ok=True)
        self.engine.write_config("encrypt", DIRS["diff_src_1bit"], DIRS["diff_cipher_1bit"])
        self.engine.run("encrypt")

    def run_baseline_decryption_stage(self):
        """Verifies bit-exact decryption reversibility of standard ciphertexts."""
        print("[PIPELINE]: Running Baseline Decryption...")
        shutil.rmtree(DIRS["decrypted"], ignore_errors=True)
        os.makedirs(DIRS["decrypted"], exist_ok=True)
        shutil.rmtree(DIRS["attacks_recovered"], ignore_errors=True)
        os.makedirs(DIRS["attacks_recovered"], exist_ok=True)
        self.engine.write_config("decrypt", DIRS["cipher"], DIRS["decrypted"])
        self.engine.run("decrypt")

    def run_structural_and_attack_stage(self, plain_files: List[str], report):
        """Computes statistical metrics (entropy, correlation, differential) and stages tampering attacks."""
        print("[PIPELINE]: Running Structural Cryptographic Audit...")
        random_crop_config = CropConfiguration(mode="random", num_boxes=15, size_range=(20, 150))
        shutil.rmtree(DIRS["attacks_staged"], ignore_errors=True)
        os.makedirs(DIRS["attacks_staged"], exist_ok=True)

        def _safe_copy(src: str, dst: str):
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except OSError:
                pass
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)

        report.write("=" * 60 + "\n")
        report.write("  PER-FRAME STRUCTURAL AUDIT\n")
        report.write("=" * 60 + "\n")

        for plain_path in plain_files:
            fname = os.path.basename(plain_path)
            c_path = get_actual_filename(DIRS["cipher"], fname)
            diff_50_path = get_actual_filename(DIRS["diff_cipher"], fname)
            diff_1_path  = get_actual_filename(DIRS["diff_cipher_1bit"], fname)

            if not os.path.exists(c_path):
                continue

            suite = CipherAuditSuite(plain_path, c_path)
            stem = os.path.splitext(fname)[0]

            # 1. Shannon Entropy
            g_ent = suite.run_entropy_audit(
                plot_out=os.path.join(DIRS["plots"], f"histogram_{stem}.png"),
                silent=True
            )
            l_ent = suite._last_local_entropy

            # 2. Adjacent Pixel Correlation
            corr = suite.run_correlation_audit(
                plot_out=os.path.join(DIRS["plots"], f"{stem}.png"),
                silent=True
            )

            # 3. Differential NPCR / UACI
            npcr_50, uaci_50 = None, None
            npcr_1,  uaci_1  = None, None
            if os.path.exists(diff_50_path):
                npcr_50 = DifferentialAnalyzer.calculate_npcr(suite.cipher, cv2.imread(diff_50_path))
                uaci_50 = DifferentialAnalyzer.calculate_uaci(suite.cipher, cv2.imread(diff_50_path))
            if os.path.exists(diff_1_path):
                npcr_1 = DifferentialAnalyzer.calculate_npcr(suite.cipher, cv2.imread(diff_1_path))
                uaci_1 = DifferentialAnalyzer.calculate_uaci(suite.cipher, cv2.imread(diff_1_path))

            # Record metrics
            report.write(f"\n  [ {fname} ]\n")
            report.write(f"  {'Global Entropy':<28}: {g_ent:.5f} / 8.0\n")
            report.write(f"  {'Local Entropy':<28}: {l_ent:.5f}\n")
            report.write(f"  {'Correlation (H) Plain|Cipher':<28}: {corr['h_plain']:8.5f} | {corr['h_cipher']:8.5f}\n")
            report.write(f"  {'Correlation (V) Plain|Cipher':<28}: {corr['v_plain']:8.5f} | {corr['v_cipher']:8.5f}\n")
            report.write(f"  {'Correlation (D) Plain|Cipher':<28}: {corr['d_plain']:8.5f} | {corr['d_cipher']:8.5f}\n")
            if npcr_50 is not None:
                report.write(f"  {'Differential NPCR (50-bit)':<28}: {npcr_50:.4f}%\n")
                report.write(f"  {'Differential UACI (50-bit)':<28}: {uaci_50:.4f}%\n")
            if npcr_1 is not None:
                report.write(f"  {'Differential NPCR (1-bit)':<28}: {npcr_1:.4f}%\n")
                report.write(f"  {'Differential UACI (1-bit)':<28}: {uaci_1:.4f}%\n")

            # Stage robustness tampering
            crop_out = os.path.join(DIRS["attacks_staged"], f"crop_{fname}")
            noise_out = os.path.join(DIRS["attacks_staged"], f"noise_{fname}")
            RobustnessAnalyzer.stage_advanced_cropping(suite.cipher, random_crop_config, crop_out)
            RobustnessAnalyzer.stage_noise_attack(suite.cipher, noise_out, density=0.05)

            aux_src = get_actual_filename(DIRS["cipher"], f"aux_{fname}")
            if os.path.exists(aux_src):
                _safe_copy(aux_src, os.path.join(DIRS["attacks_staged"], f"aux_crop_{fname}"))
                _safe_copy(aux_src, os.path.join(DIRS["attacks_staged"], f"aux_noise_{fname}"))

    def run_robustness_recovery_stage(self, plain_files: List[str], report):
        """Decrypts tampered ciphertexts and measures reconstruction fidelity (PSNR, SSIM)."""
        print("[PIPELINE]: Running Robustness Attack Recovery...")
        self.engine.write_config("decrypt", DIRS["attacks_staged"], DIRS["attacks_recovered"])
        self.engine.run("decrypt")

        report.write("\n" + "=" * 60 + "\n")
        report.write("  ROBUSTNESS & FAULT TOLERANCE\n")
        report.write("=" * 60 + "\n")

        for plain_path in plain_files:
            fname = os.path.basename(plain_path)
            rec_crop  = get_actual_filename(DIRS["attacks_recovered"], f"crop_{fname}")
            rec_noise = get_actual_filename(DIRS["attacks_recovered"], f"noise_{fname}")
            if os.path.exists(rec_crop) and os.path.exists(rec_noise):
                c_path = get_actual_filename(DIRS["cipher"], fname)
                suite = CipherAuditSuite(plain_path, c_path)
                psnr_c, ssim_c = RobustnessAnalyzer.evaluate_quality(suite.plain, cv2.imread(rec_crop))
                psnr_n, ssim_n = RobustnessAnalyzer.evaluate_quality(suite.plain, cv2.imread(rec_noise))
                report.write(f"\n  [ {fname} ]\n")
                report.write(f"  {'Crop Attack  PSNR':<28}: {psnr_c:5.2f} dB\n")
                report.write(f"  {'Crop Attack  SSIM':<28}: {ssim_c:.4f}\n")
                report.write(f"  {'Noise Attack PSNR':<28}: {psnr_n:5.2f} dB\n")
                report.write(f"  {'Noise Attack SSIM':<28}: {ssim_n:.4f}\n")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    print("[BOOT]: Initializing Automated Cipher Master Controller...")

    AssetManager.initialize_workspace()
    AssetManager.generate_zero_entropy_vectors()

def main():
    parser = argparse.ArgumentParser(
        description="DNA Cipher Autonomous Security Audit & Verification Suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "binary",
        nargs="?",
        default=None,
        help="Path to compiled CUDA cipher engine executable (e.g. DNA_CipherEngine.exe, DNA_CipherEngine_2D.exe)."
    )
    parser.add_argument(
        "-b", "--binary-path",
        dest="flag_binary",
        default=None,
        help="Explicit flag to specify cipher engine binary path."
    )

    args = parser.parse_args()
    chosen_binary = args.flag_binary or args.binary

    print("[BOOT]: Initializing Automated Cipher Master Controller...")

    AssetManager.initialize_workspace()
    AssetManager.generate_zero_entropy_vectors()

    plain_files = AssetManager.standardize_plaintexts()
    if not plain_files:
        print("[FATAL]: No valid plaintext frames found in assets/plainText. Exiting.")
        sys.exit(0)

    print(f"[ASSETS]: Normalized {len(plain_files)} test frames to {TARGET_W}x{TARGET_H}.")

    # Generate differential test vectors
    AssetManager.synthesize_differential_sources(plain_files)

    # Initialize Engine Controller with specified binary or automatic candidate discovery
    engine = CipherEngineController(chosen_binary)
    pipeline = AuditPipeline(engine)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(DIRS["reports"], f"audit_{timestamp}.txt")

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("=" * 60 + "\n")
        report.write("  DNA CIPHER SECURITY AUDIT REPORT\n")
        report.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"  Frames    : {len(plain_files)}\n")
        report.write(f"  Target    : {TARGET_W}x{TARGET_H}\n")
        report.write(f"  Engine    : {engine.binary_path}\n")
        report.write("=" * 60 + "\n\n")

        # 1. Key Sensitivity
        pipeline.run_key_avalanche_stage(plain_files, report)

        # 2. Differential Encryption
        pipeline.run_differential_encryption_stage()

        # 3. Baseline Decryption
        pipeline.run_baseline_decryption_stage()

        # 4. Structural Audit & Attack Staging
        pipeline.run_structural_and_attack_stage(plain_files, report)

        # 5. Robustness Recovery
        pipeline.run_robustness_recovery_stage(plain_files, report)

        report.write("\n" + "=" * 60 + "\n")
        report.write("  END OF REPORT\n")
        report.write("=" * 60 + "\n")

    print(f"\n[SUCCESS]: Full audit complete. Comprehensive report saved to -> '{report_path}'")


if __name__ == "__main__":
    main()