import os
import sys
import glob
import subprocess
import cv2
import random
import numpy as np # Added for zero-entropy generation
import shutil
from pythonAuditCipher.orchestrator import CipherAuditSuite
from pythonAuditCipher.robustness import CropConfiguration, CropBox 
from pythonAuditCipher.differential import DifferentialAnalyzer

BASE = "assets"
DIRS = {
    "plain": f"{BASE}/plainText",
    "cipher": f"{BASE}/cipherText",
    "decrypted": f"{BASE}/decryptedCipherText",
    "diff_src": f"{BASE}/test_differential/diff_src",
    "diff_cipher": f"{BASE}/test_differential/diff_cipher",
    "attacks_staged": f"{BASE}/test_robustness/attacks_staged",
    "attacks_recovered": f"{BASE}/test_robustness/attacks_recovered"
}

TARGET_W, TARGET_H = 1920, 1080

def build_architecture():
    for path in DIRS.values(): os.makedirs(path, exist_ok=True)

def generate_zero_entropy_vectors(input_dir: str):
    """
    Synthesizes pure uniform images directly into the input pipeline.
    """
    print("\n[BOOT]: Synthesizing Zero-Entropy Test Vectors...")
    black_path = os.path.join(input_dir, "test_00_pure_black.png")
    white_path = os.path.join(input_dir, "test_01_pure_white.png")
    
    # 3-channel (BGR) uniform matrices
    cv2.imwrite(black_path, np.full((TARGET_H, TARGET_W, 3), (0, 0, 0), dtype=np.uint8))
    cv2.imwrite(white_path, np.full((TARGET_H, TARGET_W, 3), (255, 255, 255), dtype=np.uint8))

def standardize_assets():
    plain_files = glob.glob(f"{DIRS['plain']}/*.*")
    if not plain_files: return False
    valid_count = 0
    for f in plain_files:
        img = cv2.imread(f)
        if img is None: continue
        h, w = img.shape[:2]
        if h < TARGET_H or w < TARGET_W: os.remove(f); continue
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
    """
    Generates the chaotic master keys. 
    If tweak_chen is True, it alters one parameter by a microscopic margin.
    """
    chen_k1, chen_k2, chen_k3 = 35.0, 3.0, 28.0
    chen_x, chen_y, chen_z = 0.1234567, 0.5432198, 0.9876543
    
    if tweak_chen:
        chen_x += 0.0000001 # The Avalanche Tweak!

    lor_a, lor_b, lor_c, lor_r = 10.0, (8.0 / 3.0), 46.0, 2.0
    lor_x, lor_y, lor_z, lor_w = 12.0, 0.7194113, 0.8156727, 0.2946892

    with open(filepath, "w") as f:
        f.write("[CHEN]\n")
        f.write(f"{chen_k1} {chen_k2} {chen_k3} 1000 {chen_x:.15f} {chen_y} {chen_z}\n")
        f.write("[LORENZ]\n")
        f.write(f"{lor_a} {lor_b} {lor_c} {lor_r} {lor_x} 1500 {lor_y} {lor_z} {lor_w} 0.4389124\n")

def run_key_sensitivity_test(plain_files):
    """
    Runs the C++ engine twice to test Avalanche effect, then restores 
    the standard outputs so the rest of the pipeline is completely unaffected.
    """
    print("\n[HOST]: Executing Key Sensitivity Test (Avalanche on Keys)...")
    
    # 1. Standard Run
    write_cpp_config("encrypt", DIRS['plain'], DIRS['cipher'])
    generate_master_keys("engine_keys.txt", tweak_chen=False)
    run_cpp_engine("encrypt")
    
    # 2. Quarantine the Base Output
    base_backup_dir = f"{BASE}/cipherText_Base_Backup"
    if os.path.exists(base_backup_dir): shutil.rmtree(base_backup_dir)
    shutil.copytree(DIRS['cipher'], base_backup_dir)
    
    # 3. Mutated Run
    generate_master_keys("engine_keys.txt", tweak_chen=True)
    run_cpp_engine("encrypt") # This silently overwrites DIRS['cipher']
    
    # 4. Evaluate Differences
    print("\n=== KEY SENSITIVITY TEST RESULTS ===")
    for plain_path in plain_files:
        filename = os.path.basename(plain_path)
        img_base = cv2.imread(os.path.join(base_backup_dir, filename))
        img_mutated = cv2.imread(os.path.join(DIRS['cipher'], filename))
        
        if img_base is not None and img_mutated is not None:
            npcr = DifferentialAnalyzer.calculate_npcr(img_base, img_mutated)
            uaci = DifferentialAnalyzer.calculate_uaci(img_base, img_mutated)
            print(f"File: {filename} | NPCR: {npcr:.4f}% | UACI: {uaci:.4f}%")

    # 5. Restore Architecture
    shutil.rmtree(DIRS['cipher'])
    os.rename(base_backup_dir, DIRS['cipher'])
    generate_master_keys("engine_keys.txt", tweak_chen=False) # Reset key
    print("  -> Baseline architecture restored. Continuing pipeline...")

def main():
    print("[BOOT]: Initializing Automated Cipher Master Controller...")
    build_architecture()

    # PRE-COMPUTE: Inject Zero-Entropy vectors before standardization
    generate_zero_entropy_vectors(DIRS['plain'])

    if not standardize_assets():
        sys.exit(0)

    plain_files = sorted(glob.glob(f"{DIRS['plain']}/*.png"))

    FLIPS_PER_IMAGE = 50
    for plain_path in plain_files:
        diff_path = os.path.join(DIRS['diff_src'], os.path.basename(plain_path))
        if not os.path.exists(diff_path):
            img = cv2.imread(plain_path)
            h, w, c = img.shape
            
            for _ in range(FLIPS_PER_IMAGE):
                rx = random.randint(0, w - 1)
                ry = random.randint(0, h - 1)
                rc = random.randint(0, c - 1)
                bit_pos = random.randint(0, 7)
                img[ry, rx, rc] ^= (1 << bit_pos) 
                
            cv2.imwrite(diff_path, img)

    # 2. RUN ENCRYPTION & KEY SENSITIVITY TEST
    # This single function call handles the standard plain->cipher encryption,
    # does the mutated avalanche test, and leaves the directories perfectly clean.
    run_key_sensitivity_test(plain_files)

    write_cpp_config("encrypt", DIRS['diff_src'], DIRS['diff_cipher'])
    run_cpp_engine("encrypt")

    # 3. RUN BASELINE DECRYPTION
    write_cpp_config("decrypt", DIRS['cipher'], DIRS['decrypted'])
    run_cpp_engine("decrypt")

    # 4. RUN AUDIT & STAGE ATTACKS
    print("\n[HOST]: Beginning Cryptographic Audit...")
    
    random_crop_config = CropConfiguration(mode="random", num_boxes=15, size_range=(20, 150))
    custom_crop_config = CropConfiguration(mode="custom", custom_boxes=[
        CropBox(x=100, y=100, w=500, h=500), 
        CropBox(x=1500, y=800, w=200, h=200) 
    ])
    
    active_crop_config = random_crop_config 

    for idx, plain_path in enumerate(plain_files):
        filename = os.path.basename(plain_path)
        
        cipher_path = os.path.join(DIRS['cipher'], filename)
        diff_cipher_path = os.path.join(DIRS['diff_cipher'], filename)
        
        if not os.path.exists(cipher_path): continue

        suite = CipherAuditSuite(plain_path, cipher_path)
        print(f"\n=== AUDITING: {filename} ===")
        
        suite.run_entropy_audit(plot_out=f"{BASE}/histogram_{filename}.png")
        suite.run_correlation_audit(plot_out=f"{BASE}/correlation_{filename}.png")
        suite.run_differential_audit(diff_cipher_path)

        crop_target = os.path.join(DIRS['attacks_staged'], f"crop_{filename}")
        noise_target = os.path.join(DIRS['attacks_staged'], f"noise_{filename}")
        
        from pythonAuditCipher.robustness import RobustnessAnalyzer
        RobustnessAnalyzer.stage_advanced_cropping(suite.cipher, active_crop_config, crop_target)
        RobustnessAnalyzer.stage_noise_attack(suite.cipher, noise_target, density=0.05)

        aux_src = os.path.join(DIRS['cipher'], f"aux_{filename}")
        # Only copy aux files if they exist (handling standard vs edge cases)
        if os.path.exists(aux_src):
            shutil.copy2(aux_src, os.path.join(DIRS['attacks_staged'], f"aux_crop_{filename}"))
            shutil.copy2(aux_src, os.path.join(DIRS['attacks_staged'], f"aux_noise_{filename}"))

    # 5. RUN ROBUSTNESS DECRYPTION
    write_cpp_config("decrypt", DIRS['attacks_staged'], DIRS['attacks_recovered'])
    run_cpp_engine("decrypt")

    # SYNC BRIDGE
    staged_files = sorted(glob.glob(f"{DIRS['attacks_staged']}/*.png"))
    recovered_files = glob.glob(f"{DIRS['attacks_recovered']}/decrypted_frame_*.png")
    recovered_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    for staged, rec in zip(staged_files, recovered_files):
        target_name = os.path.join(DIRS['attacks_recovered'], f"decrypted_{os.path.basename(staged)}")
        if os.path.exists(rec): os.replace(rec, target_name)

    # 6. EVALUATE RECOVERY
    print("\n[HOST]: Evaluating Cipher Robustness & Fault Tolerance...")
    for plain_path in plain_files:
        filename = os.path.basename(plain_path)
        rec_crop = os.path.join(DIRS['attacks_recovered'], f"decrypted_crop_{filename}")
        rec_noise = os.path.join(DIRS['attacks_recovered'], f"decrypted_noise_{filename}")
        
        cipher_path = os.path.join(DIRS['cipher'], filename)
        
        if os.path.exists(rec_crop) and os.path.exists(rec_noise):
            suite = CipherAuditSuite(plain_path, cipher_path) 
            print(f"\n=== RECOVERY SCORES FOR: {filename} ===")
            suite.verify_recovery(rec_crop, rec_noise)

    print("\n[SUCCESS]: Automated Pipeline Complete.")

if __name__ == "__main__":
    main()