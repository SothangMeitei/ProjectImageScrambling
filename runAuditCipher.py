import os
import sys
from pythonAuditCipher.orchestrator import CipherAuditSuite

def main():
    # File Paths
    PLAIN               = "assets/frame_0001.png"
    CIPHER_STANDARD     = "outputs/encrypted_frame_0.png"
    CIPHER_DIFF         = "benchmark_outputs/encrypted_frame_0.png" # Assuming this holds the diff cipher
    
    CROP_TARGET         = "outputs/attack_cropped.png"
    NOISE_TARGET        = "outputs/attack_noisy.png"
    
    # Ensure outputs exist
    if not os.path.exists(CIPHER_STANDARD):
        print("Run the C++ engine first: ./engine encrypt")
        sys.exit(1)

    print("\n[BOOT]: Initializing Modular Audit Suite...")
    suite = CipherAuditSuite(PLAIN, CIPHER_STANDARD)

    print("\n--- 1. ENTROPY ANALYSIS ---")
    suite.run_entropy_audit()

    print("\n--- 2. SPATIAL CORRELATION ---")
    suite.run_correlation_audit(plot_out="outputs/correlation_scatter.png")

    if os.path.exists(CIPHER_DIFF):
        print("\n--- 3. DIFFERENTIAL CRYPTANALYSIS ---")
        suite.run_differential_audit(CIPHER_DIFF)
    else:
        print("\n[!] Differential skipped (Requires diff ciphertext)")

    print("\n--- 4. STAGING ROBUSTNESS ATTACKS ---")
    suite.stage_attacks(CROP_TARGET, NOISE_TARGET)
    
    # --- Recovery Verification Block ---
    # Once you run the cropped/noisy ciphers through your C++ engine,
    # point these paths to the C++ decryption outputs and uncomment below.
    
    print("\n--- 5. EVALUATING ROBUSTNESS RECOVERY ---")
    REC_CROP = "decrypted_outputs/recovered_cropped.png"
    REC_NOISE = "decrypted_outputs/recovered_noisy.png"
    if os.path.exists(REC_CROP) and os.path.exists(REC_NOISE):
        suite.verify_recovery(REC_CROP, REC_NOISE)
    

if __name__ == "__main__":
    main()