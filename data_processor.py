import os
import tarfile
import pandas as pd
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.config import Config

DATA_DIR = os.path.join(os.path.dirname(__file__), "BuildingsBench_Data")
TAR_FILE = os.path.join(DATA_DIR, "BuildingsBench.tar.gz")

def download_and_extract_buildings_bench(progress_callback=None):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    extracted_path = os.path.join(DATA_DIR, "BuildingsBench")
    
    # Download
    s3 = boto3.client('s3', region_name='us-west-2', config=Config(signature_version=UNSIGNED))
    bucket = 'oedi-data-lake'
    
    files_to_download = [
        ('buildings-bench/v2.0.0/compressed/BuildingsBench.tar.gz', os.path.join(DATA_DIR, 'BuildingsBench.tar.gz')),
        ('buildings-bench/v2.0.0/compressed/metadata.tar.gz', os.path.join(DATA_DIR, 'metadata.tar.gz'))
    ]
    
    for key, local_file in files_to_download:
        if not os.path.exists(local_file) and not os.path.exists(os.path.join(extracted_path, "metadata", "benchmark.toml")):
            if progress_callback: progress_callback(f"Downloading {os.path.basename(local_file)}...", 0)
            # Get file size
            response = s3.head_object(Bucket=bucket, Key=key)
            total_size = int(response.get('ContentLength', 0))
            
            # Stream download
            downloaded = 0
            with open(local_file, 'wb') as f:
                resp = s3.get_object(Bucket=bucket, Key=key)
                for chunk in resp['Body'].iter_chunks(chunk_size=1024*1024*10):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress_callback(f"Downloading {os.path.basename(local_file)} ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", int((downloaded/total_size)*50))
        
        # Extract
        if os.path.exists(local_file) and not os.path.exists(os.path.join(extracted_path, "metadata", "benchmark.toml")):
            if progress_callback: progress_callback(f"Extracting {os.path.basename(local_file)}...", 50)
            with tarfile.open(local_file, 'r:gz') as tar:
                tar.extractall(path=DATA_DIR)
            
    os.environ["BUILDINGS_BENCH"] = extracted_path
    if progress_callback: progress_callback("BuildingsBench Data Ready", 100)

def generate_mock_energy_data(scenario, seq_len, n_samples):
    """
    Fallback mathematical generation replicating real energy trends
    for scenarios if device storage is exhausted.
    """
    dataX = []
    for _ in range(n_samples):
        # Base daily shape: Sine wave (low at night, peak in afternoon)
        hours = np.linspace(0, 2 * np.pi, seq_len)
        base_load = np.sin(hours - np.pi / 2) * 5 + 15  # Scaled to 10 - 20 kWh
        noise = np.random.normal(0, 0.8, seq_len)
        seq = base_load + noise
        
        if scenario == "Load Surge":
            # Induce a sharp random spike imitating heavy HVAC/Equipment usage
            surge_hour = np.random.randint(14, 18)
            seq[surge_hour] += np.random.uniform(15, 25)
            # Add secondary smaller spike
            seq[surge_hour-1] += np.random.uniform(5, 10)
            
        dataX.append(seq.reshape(-1, 1))
    return dataX

def load_scenario_data(scenario="Baseline", seq_len=24, n_samples=100, progress_callback=None):
    """
    Downloads, extracts and filters BuildingsBench evaluation data based on Scenario.
    scenario: 'Baseline' or 'Load Surge'
    Returns dataX: List of numpy arrays, each of shape [seq_len, 1]
    """
    try:
        download_and_extract_buildings_bench(progress_callback)
        
        from buildings_bench import load_pandas_dataset
        
        if progress_callback: progress_callback("Loading Pandas Dataset...", 50)
        # bdg-2:panther is one of the evaluation subsets
        generator = load_pandas_dataset('bdg-2:panther')
        
        # Generator yields building_name, dataframe
        building_name, df = next(generator)
        
        # df has a datetime index and 'load' column
        loads = df['load'].values
        
        # Chop into sequences of length seq_len
        num_seq = len(loads) // seq_len
        sequences = []
        for i in range(num_seq):
            seq = loads[i*seq_len : (i+1)*seq_len]
            if not np.any(np.isnan(seq)):
                sequences.append(seq)
                
        # Calculate variation metrics to separate scenarios
        # Variation = Max - Min
        variations = [np.max(s) - np.min(s) for s in sequences]
        
        # We sort sequences by their load variation
        sorted_pairs = sorted(zip(variations, sequences), key=lambda x: x[0])
        
        if scenario == "Baseline":
            # Baseline: normal cyclical, low variation
            selected_seqs = [seq for var, seq in sorted_pairs[:n_samples]]
        elif scenario == "Load Surge":
            # Load Surge: extreme variations
            selected_seqs = [seq for var, seq in sorted_pairs[-n_samples:]]
        else:
            selected_seqs = [seq for var, seq in sorted_pairs[n_samples:n_samples*2]]
            
        # Convert to TimeGAN expected shape
        dataX = []
        for seq in selected_seqs:
            dataX.append(seq.reshape(-1, 1))
            
        if progress_callback: progress_callback(f"Successfully loaded {len(dataX)} real {scenario} sequences.", 100)
        return dataX
        
    except Exception as e:
        if progress_callback: progress_callback(f"Storage Error, generating mathematical '{scenario}' fallback...", 100)
        print(f"Extraction failed ({e}), using math fallback generator.")
        return generate_mock_energy_data(scenario, seq_len, n_samples)


