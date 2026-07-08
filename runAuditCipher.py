import os
import sys
import glob
import subprocess
import cv2
from pythonAuditCipher.orchestrator import CipherAuditSuite

# ==============================================================================
# DIRECTORY ARCHITECTURE
# ==============================================================================
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
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

def standardize_assets():
    """Forces all assets to exactly 1080p. Discards small images. Crops large images."""
    plain_files = glob.glob(f"{DIRS['plain']}/*.*")
    if not plain_files: return False

    print(f"\n[HOST]: Enforcing {TARGET_W}x{TARGET_H} Standard Pipeline...")
    valid_count = 0
    
    for f in plain_files:
        img = cv2.imread(f)
        if img is None: continue
        
        h, w = img.shape[:2]
        if h < TARGET_H or w < TARGET_W:
            print(f"  [-] Discarded: {os.path.basename(f)} (Too small: {w}x{h})")
            os.remove(f)
            continue
            
        if h > TARGET_H or w > TARGET_W:
            print(f"  [*] Cropped  : {os.path.basename(f)} -> 1080p")
            img = img[0:TARGET_H, 0:TARGET_W]
        else:
            print(f"  [+] Verified : {os.path.basename(f)} is exactly 1080p")
            
        new_path = os.path.join(DIRS['plain'], f"frame_{valid_count}.png")
        cv2.imwrite(new_path, img)
        if new_path != f:
            os.remove(f)
        valid_count += 1
    return valid_count > 0

def write_cpp_config(mode, src_dir, out_dir):
    with open("engine_config.txt", "w") as f:
        f.write(f"[{mode.upper()}]\n{mode}_src = {src_dir}\n{mode}_out = {out_dir}\n")

def run_cpp_engine(mode):
    cmd = ["./DNA_CipherEngine.exe", mode]
    print(f"\n[HOST]: Triggering GPU Engine -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================
def main():
    print("[BOOT]: Initializing Automated Cipher Master Controller...")
    build_architecture()

    if not standardize_assets():
        print(f"\n[PAUSE]: Architecture built. Please put images into '{DIRS['plain']}' and run again.")
        sys.exit(0)

    plain_files = sorted(glob.glob(f"{DIRS['plain']}/*.png"))

    # 1. PREPARE DIFFERENTIAL ASSETS
    for plain_path in plain_files:
        diff_path = os.path.join(DIRS['diff_src'], os.path.basename(plain_path))
        if not os.path.exists(diff_path):
            img = cv2.imread(plain_path)
            img[0, 0, 0] ^= 1 
            cv2.imwrite(diff_path, img)

    # 2. RUN ENCRYPTION
    write_cpp_config("encrypt", DIRS['plain'], DIRS['cipher'])
    run_cpp_engine("encrypt")

    write_cpp_config("encrypt", DIRS['diff_src'], DIRS['diff_cipher'])
    run_cpp_engine("encrypt")

    # ---------------------------------------------------------
    # 3. RUN BASELINE DECRYPTION (This puts the normal decrypted images in your folder!)
    # ---------------------------------------------------------
    write_cpp_config("decrypt", DIRS['cipher'], DIRS['decrypted'])
    run_cpp_engine("decrypt")
    # ---------------------------------------------------------

    # 4. RUN AUDIT & STAGE ATTACKS
    print("\n[HOST]: Beginning Cryptographic Audit...")
    for idx, plain_path in enumerate(plain_files):
        filename = os.path.basename(plain_path)
        cipher_path = os.path.join(DIRS['cipher'], f"encrypted_{filename}")
        diff_cipher_path = os.path.join(DIRS['diff_cipher'], f"encrypted_{filename}")
        
        if not os.path.exists(cipher_path): continue

        suite = CipherAuditSuite(plain_path, cipher_path)
        print(f"\n=== AUDITING: {filename} ===")
        suite.run_entropy_audit()
        suite.run_correlation_audit(plot_out=f"{BASE}/correlation_{filename}.png")
        suite.run_differential_audit(diff_cipher_path)

        crop_target = os.path.join(DIRS['attacks_staged'], f"crop_{filename}")
        noise_target = os.path.join(DIRS['attacks_staged'], f"noise_{filename}")
        suite.stage_attacks(crop_target, noise_target)

    # 5. RUN ROBUSTNESS DECRYPTION
    write_cpp_config("decrypt", DIRS['attacks_staged'], DIRS['attacks_recovered'])
    run_cpp_engine("decrypt")

    # THE SYNC BRIDGE: Maps C++ robustness numerical outputs back to their attack names
    staged_files = sorted(glob.glob(f"{DIRS['attacks_staged']}/*.png"))
    recovered_files = glob.glob(f"{DIRS['attacks_recovered']}/decrypted_frame_*.png")
    
    # Sort recovered files strictly by their integer frame number
    recovered_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    for staged, rec in zip(staged_files, recovered_files):
        target_name = os.path.join(DIRS['attacks_recovered'], f"decrypted_{os.path.basename(staged)}")
        if os.path.exists(rec): # Safety check
            os.replace(rec, target_name)

    # 6. EVALUATE RECOVERY
    print("\n[HOST]: Evaluating Cipher Robustness & Fault Tolerance...")
    for plain_path in plain_files:
        filename = os.path.basename(plain_path)
        
        rec_crop = os.path.join(DIRS['attacks_recovered'], f"decrypted_crop_{filename}")
        rec_noise = os.path.join(DIRS['attacks_recovered'], f"decrypted_noise_{filename}")
        cipher_path = os.path.join(DIRS['cipher'], f"encrypted_{filename}")
        
        if os.path.exists(rec_crop) and os.path.exists(rec_noise):
            suite = CipherAuditSuite(plain_path, cipher_path) 
            print(f"\n=== RECOVERY SCORES FOR: {filename} ===")
            suite.verify_recovery(rec_crop, rec_noise)

    print("\n[SUCCESS]: Automated Pipeline Complete.")

if __name__ == "__main__":
    main()