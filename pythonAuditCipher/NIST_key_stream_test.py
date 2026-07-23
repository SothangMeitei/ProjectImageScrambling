import numpy as np
import os
from nistrng import pack_sequence, check_eligibility_all_battery, run_all_battery, SP800_22R1A_BATTERY

class NISTAnalyzer:
    """Automates the NIST SP 800-22 randomness testing suite for images and binary keystreams."""
         
    @staticmethod
    def run_suite(cipher_image: np.ndarray):
        """Runs the NIST battery on a standard ciphertext image array."""
        print("[NIST]: Unpacking ciphertext image into 1D binary stream...")
        binary_sequence = np.unpackbits(cipher_image.flatten()).astype(np.int32)
        NISTAnalyzer._execute_battery(binary_sequence, "Ciphertext Image Data")

    @staticmethod
    def run_suite_from_bin(bin_path: str):
        """Loads a processed C++ memory keystream dump from disk and runs the NIST battery."""
        if not os.path.exists(bin_path):
            print(f"[NIST ERROR]: Target keystream binary not found at: {bin_path}")
            return
        
        print(f"[NIST]: Ingesting processed hardware dump from {bin_path}...")
        
        # Reads the processed bytes directly from C++ output (no float casts, no masking)
        raw_bytes = np.fromfile(bin_path, dtype=np.uint8)
        
        # Unpack bytes into 1D binary stream (0s and 1s) for NIST
        binary_sequence = np.unpackbits(raw_bytes).astype(np.int32)
        
        NISTAnalyzer._execute_battery(binary_sequence, os.path.basename(bin_path))

    @staticmethod
    def _execute_battery(binary_sequence: np.ndarray, target_name: str):
        """Internal execution pipeline for handling the nistrng battery layout."""
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