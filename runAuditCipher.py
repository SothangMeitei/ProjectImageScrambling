import os
import sys
import glob
import subprocess
import cv2
import random
from pythonAuditCipher.orchestrator import CipherAuditSuite
from pythonAuditCipher.robustness import CropConfiguration, CropBox 

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


def main():
    print("[BOOT]: Initializing Automated Cipher Master Controller...")
    build_architecture()

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

    # 2. RUN ENCRYPTION
    write_cpp_config("encrypt", DIRS['plain'], DIRS['cipher'])
    run_cpp_engine("encrypt")

    write_cpp_config("encrypt", DIRS['diff_src'], DIRS['diff_cipher'])
    run_cpp_engine("encrypt")

    # 3. RUN BASELINE DECRYPTION
    write_cpp_config("decrypt", DIRS['cipher'], DIRS['decrypted'])
    run_cpp_engine("decrypt")

    # 4. RUN AUDIT & STAGE ATTACKS
    print("\n[HOST]: Beginning Cryptographic Audit...")
    
    # --- DEFINE OUR ATTACK CONFIGURATION ---
    # Example 1: Purely random 
    random_crop_config = CropConfiguration(mode="random", num_boxes=15, size_range=(20, 150))
    
    # Example 2: Hardcoded Custom (e.g., precise data block deletion)
    custom_crop_config = CropConfiguration(mode="custom", custom_boxes=[
        CropBox(x=100, y=100, w=500, h=500), # Giant top-left crop
        CropBox(x=1500, y=800, w=200, h=200) # Small bottom-right crop
    ])
    
    # We will use the random config for this run.
    active_crop_config = random_crop_config 
    # ---------------------------------------

    for idx, plain_path in enumerate(plain_files):
        filename = os.path.basename(plain_path)
        
        # FIX 1: Remove "encrypted_" prefix to match what C++ actually saved!
        cipher_path = os.path.join(DIRS['cipher'], filename)
        diff_cipher_path = os.path.join(DIRS['diff_cipher'], filename)
        
        if not os.path.exists(cipher_path): continue

        suite = CipherAuditSuite(plain_path, cipher_path)
        print(f"\n=== AUDITING: {filename} ===")
        
        # This will now also output the Histogram!
        suite.run_entropy_audit(plot_out=f"{BASE}/histogram_{filename}.png")
        suite.run_correlation_audit(plot_out=f"{BASE}/correlation_{filename}.png")
        suite.run_differential_audit(diff_cipher_path)

        crop_target = os.path.join(DIRS['attacks_staged'], f"crop_{filename}")
        noise_target = os.path.join(DIRS['attacks_staged'], f"noise_{filename}")
        
        # Deploy the Data-Driven Advanced Crop!
        from pythonAuditCipher.robustness import RobustnessAnalyzer
        RobustnessAnalyzer.stage_advanced_cropping(suite.cipher, active_crop_config, crop_target)
        RobustnessAnalyzer.stage_noise_attack(suite.cipher, noise_target, density=0.05)

        # FIX 2: Copy the pristine 'aux_' files into the attacks folder so C++ can use them to recover!
        import shutil
        aux_src = os.path.join(DIRS['cipher'], f"aux_{filename}")
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
        
        # FIX 1 (Continued): Remove "encrypted_" prefix
        cipher_path = os.path.join(DIRS['cipher'], filename)
        
        if os.path.exists(rec_crop) and os.path.exists(rec_noise):
            suite = CipherAuditSuite(plain_path, cipher_path) 
            print(f"\n=== RECOVERY SCORES FOR: {filename} ===")
            suite.verify_recovery(rec_crop, rec_noise)

    print("\n[SUCCESS]: Automated Pipeline Complete.")

if __name__ == "__main__":
    main()