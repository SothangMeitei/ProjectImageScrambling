import cv2
import numpy as np
import os
from scipy.stats import entropy
from sewar.full_ref import psnr, ssim


PLAIN_DIR = "assets"
CIPHER_DIR = "outputs"


# ============================================================
# ENTROPY
# ============================================================

def shannon_entropy(img):

    histogram = np.bincount(
        img.flatten(),
        minlength=256
    )

    return entropy(
        histogram,
        base=2
    )



# ============================================================
# CORRELATION
# ============================================================

def correlation(img, direction):

    if direction == "H":
        x = img[:, :-1].flatten()
        y = img[:, 1:].flatten()

    elif direction == "V":
        x = img[:-1, :].flatten()
        y = img[1:, :].flatten()

    elif direction == "D":
        x = img[:-1, :-1].flatten()
        y = img[1:, 1:].flatten()


    return np.corrcoef(x,y)[0,1]



# ============================================================
# NPCR
# ============================================================

def calculate_npcr(cipher1, cipher2):

    diff = cipher1 != cipher2

    changed_pixels = np.sum(diff)

    total_pixels = cipher1.size


    return (
        changed_pixels /
        total_pixels
    ) * 100



# ============================================================
# UACI
# ============================================================

def calculate_uaci(cipher1, cipher2):

    difference = np.abs(
        cipher1.astype(np.float32)
        -
        cipher2.astype(np.float32)
    )


    return (
        np.mean(difference)
        /
        255
    ) * 100



# ============================================================
# HISTOGRAM UNIFORMITY
# ============================================================

def histogram_variance(img):

    hist,_ = np.histogram(
        img,
        bins=256,
        range=(0,255)
    )

    return np.var(hist)



# ============================================================
# STORAGE
# ============================================================

entropy_results=[]

corr_h_results=[]
corr_v_results=[]
corr_d_results=[]

psnr_results=[]
ssim_results=[]

hist_results=[]


cipher_images=[]



# ============================================================
# PROCESS ALL FRAMES
# ============================================================


cipher_files = sorted(
    [
        f for f in os.listdir(CIPHER_DIR)
        if f.endswith(".png")
    ]
)



print("="*70)
print("        DNA CHAOTIC IMAGE CIPHER CRYPTANALYSIS")
print("="*70)



for cipher_file in cipher_files:


    frame_id = int(
        cipher_file.split("_")[-1]
        .split(".")[0]
    )


    plain_name = (
        f"frame_{frame_id+1:04d}.png"
    )


    plain_path = os.path.join(
        PLAIN_DIR,
        plain_name
    )


    cipher_path = os.path.join(
        CIPHER_DIR,
        cipher_file
    )


    plain=cv2.imread(
        plain_path,
        cv2.IMREAD_COLOR # <-- STOP SQUASHING THE CHAOS!
    )

    cipher=cv2.imread(
        cipher_path,
        cv2.IMREAD_COLOR # <-- STOP SQUASHING THE CHAOS!
    )


    if plain is None or cipher is None:

        print(
            "[SKIP]",
            cipher_file
        )

        continue



    cipher_images.append(cipher)



    H=shannon_entropy(cipher)

    entropy_results.append(H)



    corr_h_results.append(
        correlation(cipher,"H")
    )

    corr_v_results.append(
        correlation(cipher,"V")
    )

    corr_d_results.append(
        correlation(cipher,"D")
    )



    psnr_results.append(
        psnr(
            plain,
            cipher
        )
    )


    ssim_results.append(
        ssim(
            plain,
            cipher
        )[0]
    )


    hist_results.append(
        histogram_variance(cipher)
    )



    print(
        f"{cipher_file:25}"
        f"Entropy:{H:.5f}"
        f" PSNR:{psnr_results[-1]:.2f}"
        f" SSIM:{ssim_results[-1]:.5f}"
    )



# ============================================================
# SUMMARY
# ============================================================


def report(name,data):

    print(
        f"{name:<35}"
        f"Mean={np.mean(data):.6f} "
        f"Std={np.std(data):.6f} "
        f"Min={np.min(data):.6f} "
        f"Max={np.max(data):.6f}"
    )



print("\n"+"="*70)
print("                    FINAL RESULTS")
print("="*70)



report(
    "Cipher Entropy",
    entropy_results
)


report(
    "Horizontal Correlation",
    corr_h_results
)


report(
    "Vertical Correlation",
    corr_v_results
)


report(
    "Diagonal Correlation",
    corr_d_results
)


report(
    "PSNR",
    psnr_results
)


report(
    "SSIM",
    ssim_results
)


report(
    "Histogram Variance",
    hist_results
)



# ============================================================
# NPCR / UACI
# Between consecutive ciphertext frames
# ============================================================


npcr_values=[]
uaci_values=[]


for i in range(
    len(cipher_images)-1
):

    npcr_values.append(
        calculate_npcr(
            cipher_images[i],
            cipher_images[i+1]
        )
    )


    uaci_values.append(
        calculate_uaci(
            cipher_images[i],
            cipher_images[i+1]
        )
    )



print("\nDIFFERENTIAL ANALYSIS")
print("-"*70)


report(
    "NPCR (%)",
    npcr_values
)


report(
    "UACI (%)",
    uaci_values
)



print("="*70)