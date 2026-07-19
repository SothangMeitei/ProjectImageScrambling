import numpy as np
from nistrng import pack_sequence, unpack_sequence, check_eligibility_all_queries, run_all_battery, SP800_22R1A_BATTERY

class NISTAnalyzer:
    """Automates the NIST SP 800-22 randomness testing suite."""
    
    @staticmethod
    def run_suite(cipher_image: np.ndarray):
        print("[NIST]: Unpacking ciphertext into 1D binary stream...")
        
        # 1. Flatten the image and convert it into a pure stream of 1s and 0s
        binary_sequence = np.unpackbits(cipher_image.flatten())
        
        # 2. Pack the sequence for the NIST library format
        sequence_packed = pack_sequence(binary_sequence)
        
        # 3. Check which tests have enough data to run (some need millions of bits)
        eligible_battery = check_eligibility_all_queries(binary_sequence, SP800_22R1A_BATTERY)
        
        print(f"[NIST]: Executing {len(eligible_battery.keys())} eligible statistical tests...")
        
        # 4. Run the suite
        results = run_all_battery(binary_sequence, eligible_battery, False)
        
        # 5. Print out the formal Evaluation
        passed_tests = 0
        print("\n=== NIST SP 800-22 TEST RESULTS ===")
        for result, elapsed_time in results:
            status = "PASS" if result.passed else "FAIL"
            if result.passed: passed_tests += 1
            print(f"{result.name:<30} | {status} | p-value: {result.score:.5f}")
            
        print(f"\n[NIST FINAL]: Passed {passed_tests} / {len(results)} tests.")