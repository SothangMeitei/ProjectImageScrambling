import numpy as np
import os
from nistrng import pack_sequence, check_eligibility_all_battery, run_all_battery, SP800_22R1A_BATTERY

class NISTAnalyzer:
    """Automates the NIST SP 800-22 randomness testing suite for images and binary keystreams."""
         
    @staticmethod
    def run_suite(cipher_image: np.ndarray):
        """Runs the NIST battery on a standard image array."""
        print("[NIST]: Unpacking ciphertext image into 1D binary stream...")
        # FIXED: Cast to int32 to prevent Cumulative Sums overflow during random walks
        binary_sequence = np.unpackbits(cipher_image.flatten()).astype(np.int32)
        NISTAnalyzer._execute_battery(binary_sequence, "Ciphertext Image Data")

    @staticmethod
    def run_suite_from_bin(bin_path: str):
        """Loads a raw C++ memory keystream dump from disk and runs the NIST battery."""
        if not os.path.exists(bin_path):
            print(f"[NIST ERROR]: Target keystream binary not found at: {bin_path}")
            return
        
        print(f"[NIST]: Ingesting raw hardware dump from {bin_path}...")
        
        # 1. Read the raw floats directly from the C++ engine memory dump
        raw_floats = np.fromfile(bin_path, dtype=np.float32)
        
        # 2. View them as 32-bit integers to bypass Python's high-level float rounding
        raw_ints = raw_floats.view(np.uint32)
        
        # 3. Strip the rigid IEEE 754 Sign and Exponent bits. 
        # Keep ONLY the lowest 8 bits of the mantissa (the deepest chaotic fraction).
        pure_entropy_bytes = (raw_ints & 0xFF).astype(np.uint8)
        
        # 4. Unpack to binary, using int32 for mathematical safety
        binary_sequence = np.unpackbits(pure_entropy_bytes).astype(np.int32)
        
        NISTAnalyzer._execute_battery(binary_sequence, os.path.basename(bin_path))

    @staticmethod
    def _execute_battery(binary_sequence: np.ndarray, target_name: str):
        """Internal execution pipeline for handling the nistrng battery layout."""
        # OPTIMIZATION: Slice the sequence down to the NIST standard sample size
        NIST_STANDARD_SAMPLE = 1000000
        if len(binary_sequence) > NIST_STANDARD_SAMPLE:
            print(f"[NIST OPTIMIZATION]: Slicing stream from {len(binary_sequence)} down to {NIST_STANDARD_SAMPLE} bits.")
            binary_sequence = binary_sequence[:NIST_STANDARD_SAMPLE]
            
        sequence_packed = pack_sequence(binary_sequence)
        eligible_battery = check_eligibility_all_battery(binary_sequence, SP800_22R1A_BATTERY)
        
        print(f"[NIST]: Executing {len(eligible_battery.keys())} eligible statistical tests...")
        results = run_all_battery(binary_sequence, eligible_battery, False)
        
        passed_tests = 0
        print(f"\n=== NIST SP 800-22 RESULTS: {target_name} ===")
        for result, elapsed_time in results:
            status = "PASS" if result.passed else "FAIL"
            if result.passed: 
                passed_tests += 1
            print(f"  -> {result.name:<30} | {status} | p-value: {result.score:.5f}")
                 
        print(f"\n[NIST SUMMARY]: '{target_name}' Passed {passed_tests} / {len(results)} tests.")