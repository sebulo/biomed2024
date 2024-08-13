import os
import json
import numpy as np

def generate_zero_predictions(settings):
    result_dir = settings["result_dir"]
    test_list_file = os.path.join(result_dir, settings["data_set"])
    
    # Load test sample IDs
    test_ids = np.loadtxt(test_list_file, delimiter=",", dtype=str)
    
    # Generate predictions (all 0s)
    predictions = {sample_id.strip(): 0 for sample_id in test_ids}
    
    # Save predictions to test_results.json
    test_results_json = os.path.join(result_dir, "test_results.json")
    with open(test_results_json, 'w') as outfile:
        json.dump(predictions, outfile, indent=4)
    
    print(f"Zero predictions saved to {test_results_json}")

if __name__ == '__main__':
    from dtu_spine_config import DTUConfig
    import argparse

    args = argparse.ArgumentParser(description='Generate Zero Predictions')
    config = DTUConfig(args)
    if config.settings is not None:
        generate_zero_predictions(config.settings)