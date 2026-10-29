import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "BuildingsBench_Data")

# BDG-2 site CSVs are served individually from the public OEDI bucket,
# so there's no need to pull the full multi-GB BuildingsBench tarball.
S3_BASE = "https://oedi-data-lake.s3.us-west-2.amazonaws.com/buildings-bench/v2.0.0/BuildingsBench/BDG-2"
DEFAULT_SITE = "Panther"
DEFAULT_YEAR = 2017
DEFAULT_BUILDING = "Panther_office_Hannah"

def download_site_csv(site=DEFAULT_SITE, year=DEFAULT_YEAR, progress_callback=None):
    """
    Downloads one BDG-2 site CSV (~8 MB, one column per building, hourly) and caches it locally.
    Returns the local file path.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{site}_clean={year}.csv"
    local_file = os.path.join(DATA_DIR, filename)

    if not os.path.exists(local_file):
        if progress_callback: progress_callback(f"Downloading {filename}...", 0)
        tmp_file = local_file + ".part"
        with urllib.request.urlopen(f"{S3_BASE}/{filename}", timeout=60) as resp, open(tmp_file, 'wb') as f:
            total_size = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress_callback(f"Downloading {filename} ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", int((downloaded / total_size) * 50))
        os.replace(tmp_file, local_file)

    return local_file

def load_site_dataframe(site=DEFAULT_SITE, year=DEFAULT_YEAR, progress_callback=None):
    local_file = download_site_csv(site, year, progress_callback)
    return pd.read_csv(local_file, index_col=0, parse_dates=True)

def list_buildings(site=DEFAULT_SITE, year=DEFAULT_YEAR):
    """Returns the building column names for a site (reads only the CSV header)."""
    local_file = download_site_csv(site, year)
    return list(pd.read_csv(local_file, index_col=0, nrows=0).columns)

def generate_mock_energy_data(scenario, seq_len, n_samples):
    """
    Fallback mathematical generation replicating real energy trends
    for scenarios if the real data can't be fetched (e.g. offline).
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

def load_scenario_data(scenario="Baseline", seq_len=24, n_samples=100, building=DEFAULT_BUILDING, progress_callback=None):
    """
    Loads one BDG-2 building's hourly load and filters daily sequences by Scenario.
    scenario: 'Baseline' or 'Load Surge'
    Returns (dataX, source): dataX is a list of numpy arrays, each of shape [seq_len, 1];
    source describes where the data came from ('real' building name or 'mock fallback').
    """
    try:
        if progress_callback: progress_callback("Loading BDG-2 site data...", 0)
        df = load_site_dataframe(progress_callback=progress_callback)
        if building not in df.columns:
            raise KeyError(f"Building '{building}' not found in site data")

        if progress_callback: progress_callback(f"Slicing {building} into daily sequences...", 60)
        loads = df[building].values

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

        # A single building-year only has ~365 days, so cap at half to keep the scenarios disjoint
        n_samples = min(n_samples, len(sorted_pairs) // 2)

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
        return dataX, building

    except Exception as e:
        if progress_callback: progress_callback(f"Real data unavailable, generating mathematical '{scenario}' fallback...", 100)
        print(f"Real data load failed ({e}), using math fallback generator.")
        return generate_mock_energy_data(scenario, seq_len, n_samples), "mock fallback"
