import cv2
import numpy as np
from scipy.stats import entropy
from sewar.full_ref import psnr, ssim

# 1. Ingest initial Plaintext frame and its corresponding Ciphertext frame contiguously
plain  = cv2.imread("assets/frame_0001.png", cv2.IMREAD_GRAYSCALE)
cipher = cv2.imread("outputs/encrypted_frame_0.png", cv2.IMREAD_GRAYSCALE)

if plain is None or cipher is None:
    print("[SYSTEM ERROR]: Could not load image buffers. Ensure the C++ engine has processed the batch!")
    exit()

print("============================================================")
print("          DNA CHAOTIC CRYPTANALYSIS EVALUATION              ")
print("============================================================\n")

# --- METRIC 1: INFORMATION ENTROPY (H) ---
# Evaluates the uniformity of the byte distribution (Theoretical maximum = 8.0)
def get_shannon_entropy(img):
    counts = np.bincount(img.flatten(), minlength=256)
    return entropy(counts, base=2)

plain_H  = get_shannon_entropy(plain)
cipher_H = get_shannon_entropy(cipher)
print(f"[ENTROPY] Plaintext Source  : {plain_H:.5f} / 8.0")
print(f"[ENTROPY] Ciphertext Output : {cipher_H:.5f} / 8.0  (IEEE Target: > 7.999)\n")

# --- METRIC 2: ADJACENT PIXEL CORRELATION COEFFICIENTS ---
# Evaluates structural rigidity against Chosen-Plaintext linear regression
def get_correlation(img, direction='H'):
    if direction == 'H':   x, y = img[:, :-1].flatten(), img[:, 1:].flatten()
    elif direction == 'V': x, y = img[:-1, :].flatten(), img[1:, :].flatten()
    elif direction == 'D': x, y = img[:-1, :-1].flatten(), img[1:, 1:].flatten()
    return np.corrcoef(x, y)[0, 1]

print(f"[CORRELATION] Plaintext Horizontal  : {get_correlation(plain, 'H'):.5f}")
print(f"[CORRELATION] Ciphertext Horizontal : {get_correlation(cipher, 'H'):.5f}  (Target: ~0.0000)")
print(f"[CORRELATION] Ciphertext Vertical   : {get_correlation(cipher, 'V'):.5f}  (Target: ~0.0000)")
print(f"[CORRELATION] Ciphertext Diagonal   : {get_correlation(cipher, 'D'):.5f}  (Target: ~0.0000)\n")

# --- METRIC 3: SIGNAL DESTRUCTION & STRUCTURAL SURVIVAL ---
# PSNR evaluates noise cloaking power; SSIM evaluates structural silhouette leakage
print(f"[DESTRUCTION] Peak Signal-to-Noise (PSNR) : {psnr(plain, cipher):.2f} dB  (Target: < 10 dB)")
print(f"[DESTRUCTION] Structural Similarity (SSIM): {ssim(plain, cipher)[0]:.5f}     (Target: ~0.0000)\n")

print("============================================================")