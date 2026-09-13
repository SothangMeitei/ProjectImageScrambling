import os
import sys
import glob
import subprocess
import cv2
import random
import numpy as np 
import shutil
from datetime import datetime
from pythonAuditCipher.orchestrator import CipherAuditSuite
from pythonAuditCipher.robustness import CropConfiguration, RobustnessAnalyzer
from pythonAuditCipher.differential import DifferentialAnalyzer

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
    "plots": f"{BASE}/reports/plots"
}

TARGET_W, TARGET_H = 1920, 1080

def build_architecture():
    for path in DIRS.values(): os.makedirs(path, exist_ok=True)

def generate_zero_entropy_vectors():
    """Synthesizes pure uniform images into an isolated test folder (not plainText)."""
    print("\n[BOOT]: Synthesizing Zero-Entropy Test Vectors...")
    os.makedirs(DIRS['zero_entropy'], exist_ok=True)
    black_path = os.path.join(DIRS['zero_entropy'], "pure_black.png")
    white_path = os.path.join(DIRS['zero_entropy'], "pure_white.png")
    
    if not os.path.exists(black_path):
        cv2.imwrite(black_path, np.full((TARGET_H, TARGET_W, 3), (0, 0, 0), dtype=np.uint8))
    if not os.path.exists(white_path):
        cv2.imwrite(white_path, np.full((TARGET_H, TARGET_W, 3), (255, 255, 255), dtype=np.uint8))

def standardize_assets():
    plain_files = glob.glob(f"{DIRS['plain']}/*.*")
    if not plain_files: return False
    valid_count = 0
    for f in plain_files:
        img = cv2.imread(f)
        if img is None: continue
        h, w = img.shape[:2]
        if h < TARGET_H or w < TARGET_W: 
            print(f"[WARNING]: Skipping and deleting {f} - below {TARGET_W}x{TARGET_H}")
            os.remove(f)
            continue
        if h > TARGET_H or w > TARGET_W: img = img[0:TARGET_H, 0:TARGET_W]
        new_path = os.path.join(DIRS['plain'], f"frame_{valid_count}.png")
        cv2.imwrite(new_path, img)
        if new_path != f: os.remove(f)
        valid_count += 1
    return valid_count > 0

def write_cpp_config(mode, src_dir, out_dir):
    with open("engine_config.txt", "w") as f:
        f.write(f"[{mode.upper()}]\n{mode}_src = {src_dir}\n{mode}_out = {out_dir}\n")

def run_cpp_engine(mode):
    cmd = ["./DNA_CipherEngine.exe", mode]
    subprocess.run(cmd, check=True)

def generate_master_keys(filepath="engine_keys.txt", tweak_chen=False):
    """Generates the chaotic master keys with massive burn-in for true divergence."""
    chen_k1, chen_k2, chen_k3 = 35.0, 3.0, 28.0
    chen_x, chen_y, chen_z = 0.1234567, 0.5432198, 0.9876543
    
    if tweak_chen:
        chen_x += 0.000001 

    lor_a, lor_b, lor_c, lor_r = 10.0, (8.0 / 3.0), 46.0, 2.0
    lor_x, lor_y, lor_z, lor_w = 12.0, 0.7194113, 0.8156727, 0.2946892

    with open(filepath, "w") as f:
        f.write("[CHEN]\n")
        f.write(f"{chen_k1} {chen_k2} {chen_k3} 500000 {chen_x:.15f} {chen_y} {chen_z}\n")
        f.write("[LORENZ]\n")
        f.write(f"{lor_a} {lor_b} {lor_c} {lor_r} {lor_x} 500000 {lor_y} {lor_z} {lor_w} 0.4389124\n")

def run_key_sensitivity_test(plain_files, report):
    """Runs the C++ engine twice to test Avalanche effect and restores output layout."""
    print("[PIPELINE]: Running Key Sensitivity Test...")
    
    write_cpp_config("encrypt", DIRS['plain'], DIRS['cipher'])
    generate_master_keys("engine_keys.txt", tweak_chen=False)
    run_cpp_engine("encrypt")
    
    base_backup_dir = f"{BASE}/cipherText_Base_Backup"
    if os.path.exists(base_backup_dir): shutil.rmtree(base_backup_dir)
    shutil.copytree(DIRS['cipher'], base_backup_dir)
    
    # [CRITICAL FIX]: Delete old cipherText files to guarantee Windows doesn't block overwriting
    shutil.rmtree(DIRS['cipher'], ignore_errors=True)
    os.makedirs(DIRS['cipher'], exist_ok=True)
    
    try:
        generate_master_keys("engine_keys.txt", tweak_chen=True)
        run_cpp_engine("encrypt") 
        
        report.write("=" * 60 + "\n")
        report.write("  KEY SENSITIVITY TEST  (Avalanche Effect on Keys)\n")
        report.write("  Key Tweak: Chen x0 += 0.000001\n")
        report.write("  Expected : NPCR > 99.6% | UACI ~ 33.4%\n")
        report.write("=" * 60 + "\n")
        report.write(f"  {'File':<20} {'NPCR':>10} {'UACI':>10}\n")
        report.write("  " + "-" * 44 + "\n")
        
        npcr_vals, uaci_vals = [], []
        for plain_path in plain_files:
            filename = os.path.basename(plain_path)
            img_base    = cv2.imread(os.path.join(base_backup_dir, filename))
            img_mutated = cv2.imread(os.path.join(DIRS['cipher'], filename))
            if img_base is not None and img_mutated is not None:
                npcr = DifferentialAnalyzer.calculate_npcr(img_base, img_mutated)
                uaci = DifferentialAnalyzer.calculate_uaci(img_base, img_mutated)
                npcr_vals.append(npcr)
                uaci_vals.append(uaci)
                report.write(f"  {filename:<20} {npcr:>9.4f}% {uaci:>9.4f}%\n")
        
        if npcr_vals:
            report.write("  " + "-" * 44 + "\n")
            report.write(f"  {'AVERAGE':<20} {sum(npcr_vals)/len(npcr_vals):>9.4f}% {sum(uaci_vals)/len(uaci_vals):>9.4f}%\n")
        report.write("\n")
        
    finally:
        shutil.rmtree(DIRS['cipher'], ignore_errors=True)
        os.rename(base_backup_dir, DIRS['cipher'])
        generate_master_keys("engine_keys.txt", tweak_chen=False) 
        print("[PIPELINE]: Baseline architecture restored.")

def main():
    print("[BOOT]: Initializing Automated Cipher Master Controller...")
    build_architecture()

    generate_zero_entropy_vectors()

    if not standardize_assets():
        sys.exit(0)

    plain_files = sorted(glob.glob(f"{DIRS['plain']}/*.png"))

    # --- Prepare diff source vectors freshly from current plainText ---
    shutil.rmtree(DIRS['diff_src'], ignore_errors=True)
    os.makedirs(DIRS['diff_src'], exist_ok=True)
    shutil.rmtree(DIRS['diff_src_1bit'], ignore_errors=True)
    os.makedirs(DIRS['diff_src_1bit'], exist_ok=True)

    FLIPS_PER_IMAGE = 50
    for plain_path in plain_files:
        diff_path       = os.path.join(DIRS['diff_src'],       os.path.basename(plain_path))
        diff_path_1bit  = os.path.join(DIRS['diff_src_1bit'],  os.path.basename(plain_path))
        
        img = cv2.imread(plain_path)
        if img is not None:
            h, w, c = img.shape
            # 50-bit flips
            img_50 = img.copy()
            for _ in range(FLIPS_PER_IMAGE):
                img_50[random.randint(0, h-1), random.randint(0, w-1), random.randint(0, c-1)] ^= (1 << random.randint(0, 7))
            cv2.imwrite(diff_path, img_50)
            
            # 1-bit flip
            img_1 = img.copy()
            img_1[random.randint(0, h-1), random.randint(0, w-1), random.randint(0, c-1)] ^= (1 << random.randint(0, 7))
            cv2.imwrite(diff_path_1bit, img_1)

    # --- Open the report file ---
    os.makedirs(f"{BASE}/reports", exist_ok=True)
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = f"{BASE}/reports/audit_{timestamp}.txt"
    
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("=" * 60 + "\n")
        report.write("  DNA CIPHER SECURITY AUDIT REPORT\n")
        report.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"  Frames    : {len(plain_files)}\n")
        report.write(f"  Target    : {TARGET_W}x{TARGET_H}\n")
        report.write("=" * 60 + "\n\n")

        # 1. KEY SENSITIVITY TEST
        run_key_sensitivity_test(plain_files, report)

        # 2. DIFFERENTIAL CRYPTANALYSIS
        print("[PIPELINE]: Running Differential Cryptanalysis (50-bit flip)...")
        shutil.rmtree(DIRS['diff_cipher'], ignore_errors=True)
        os.makedirs(DIRS['diff_cipher'], exist_ok=True)
        write_cpp_config("encrypt", DIRS['diff_src'], DIRS['diff_cipher'])
        run_cpp_engine("encrypt")
        
        print("[PIPELINE]: Running Differential Cryptanalysis (1-bit flip)...")
        shutil.rmtree(DIRS['diff_cipher_1bit'], ignore_errors=True)
        os.makedirs(DIRS['diff_cipher_1bit'], exist_ok=True)
        write_cpp_config("encrypt", DIRS['diff_src_1bit'], DIRS['diff_cipher_1bit'])
        run_cpp_engine("encrypt")

        # 3. BASELINE DECRYPTION
        print("[PIPELINE]: Running Baseline Decryption...")
        shutil.rmtree(DIRS['decrypted'], ignore_errors=True)
        os.makedirs(DIRS['decrypted'], exist_ok=True)
        shutil.rmtree(DIRS['attacks_recovered'], ignore_errors=True)
        os.makedirs(DIRS['attacks_recovered'], exist_ok=True)
        write_cpp_config("decrypt", DIRS['cipher'], DIRS['decrypted'])
        run_cpp_engine("decrypt")

        # 4. PER-FRAME STRUCTURAL AUDIT
        print("[PIPELINE]: Running Structural Cryptographic Audit...")
        random_crop_config = CropConfiguration(mode="random", num_boxes=15, size_range=(20, 150))
        shutil.rmtree(DIRS['attacks_staged'], ignore_errors=True)
        os.makedirs(DIRS['attacks_staged'], exist_ok=True)

        def safe_copy(src, dst):
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
            filename            = os.path.basename(plain_path)
            cipher_path         = os.path.join(DIRS['cipher'],         filename)
            diff_cipher_path    = os.path.join(DIRS['diff_cipher'],     filename)
            diff_cipher_1bit    = os.path.join(DIRS['diff_cipher_1bit'],filename)
            
            if not os.path.exists(cipher_path):
                continue

            suite = CipherAuditSuite(plain_path, cipher_path)
            plot_stem = os.path.splitext(filename)[0]

            # --- Entropy ---
            g_ent = suite.run_entropy_audit(
                plot_out=os.path.join(DIRS['plots'], f"histogram_{plot_stem}.png"),
                silent=True
            )
            l_ent = suite._last_local_entropy

            # --- Correlation ---
            corr = suite.run_correlation_audit(
                plot_out=os.path.join(DIRS['plots'], f"{plot_stem}.png"),
                silent=True
            )

            # --- Differential ---
            npcr_50, uaci_50   = None, None
            npcr_1,  uaci_1    = None, None
            if os.path.exists(diff_cipher_path):
                npcr_50 = DifferentialAnalyzer.calculate_npcr(suite.cipher, cv2.imread(diff_cipher_path))
                uaci_50 = DifferentialAnalyzer.calculate_uaci(suite.cipher, cv2.imread(diff_cipher_path))
            if os.path.exists(diff_cipher_1bit):
                npcr_1  = DifferentialAnalyzer.calculate_npcr(suite.cipher, cv2.imread(diff_cipher_1bit))
                uaci_1  = DifferentialAnalyzer.calculate_uaci(suite.cipher, cv2.imread(diff_cipher_1bit))

            # --- Write to report ---
            report.write(f"\n  [ {filename} ]\n")
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

            # --- Stage attacks ---
            crop_target  = os.path.join(DIRS['attacks_staged'], f"crop_{filename}")
            noise_target = os.path.join(DIRS['attacks_staged'], f"noise_{filename}")
            RobustnessAnalyzer.stage_advanced_cropping(suite.cipher, random_crop_config, crop_target)
            RobustnessAnalyzer.stage_noise_attack(suite.cipher, noise_target, density=0.05)
            aux_src = os.path.join(DIRS['cipher'], f"aux_{filename}")
            if os.path.exists(aux_src):
                safe_copy(aux_src, os.path.join(DIRS['attacks_staged'], f"aux_crop_{filename}"))
                safe_copy(aux_src, os.path.join(DIRS['attacks_staged'], f"aux_noise_{filename}"))

        # 5. ROBUSTNESS (ATTACK RECOVERY)
        print("[PIPELINE]: Running Robustness Attack Recovery...")
        write_cpp_config("decrypt", DIRS['attacks_staged'], DIRS['attacks_recovered'])
        run_cpp_engine("decrypt")

        report.write("\n" + "=" * 60 + "\n")
        report.write("  ROBUSTNESS & FAULT TOLERANCE\n")
        report.write("=" * 60 + "\n")

        for plain_path in plain_files:
            filename  = os.path.basename(plain_path)
            rec_crop  = os.path.join(DIRS['attacks_recovered'], f"crop_{filename}")
            rec_noise = os.path.join(DIRS['attacks_recovered'], f"noise_{filename}")
            if os.path.exists(rec_crop) and os.path.exists(rec_noise):
                cipher_path = os.path.join(DIRS['cipher'], filename)
                suite = CipherAuditSuite(plain_path, cipher_path)
                psnr_c, ssim_c = RobustnessAnalyzer.evaluate_quality(suite.plain, cv2.imread(rec_crop))
                psnr_n, ssim_n = RobustnessAnalyzer.evaluate_quality(suite.plain, cv2.imread(rec_noise))
                report.write(f"\n  [ {filename} ]\n")
                report.write(f"  {'Crop Attack  PSNR':<28}: {psnr_c:5.2f} dB\n")
                report.write(f"  {'Crop Attack  SSIM':<28}: {ssim_c:.4f}\n")
                report.write(f"  {'Noise Attack PSNR':<28}: {psnr_n:5.2f} dB\n")
                report.write(f"  {'Noise Attack SSIM':<28}: {ssim_n:.4f}\n")

        report.write("\n" + "=" * 60 + "\n")
        report.write("  END OF REPORT\n")
        report.write("=" * 60 + "\n")

    print(f"\n[SUCCESS]: Audit complete. Report saved to -> '{report_path}'")

if __name__ == "__main__":
    main()