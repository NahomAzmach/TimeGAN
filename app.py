import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from timegan_core import tgan
import data_processor

st.set_page_config(page_title="TimeGAN: Synthetic Data Generator", page_icon="📈", layout="wide")

st.title("TimeGAN: Synthetic Energy Data Generator")
st.markdown("""
This application implements the  **Time-series Generative Adversarial Networks (TimeGAN)**.
It generates realistic time-series data while preserving the temporal dynamics (e.g. Load Surge) inherent in the training data using a **Stepwise Supervised Loss** via the Supervisor network.
""")

st.sidebar.header("Configuration")
scenario = st.sidebar.selectbox("Scenario Description", ["Baseline", "Load Surge"], help="Select the real-world temporal pattern you want to synthesize.")

iterations = st.sidebar.slider("Training Iterations", 100, 5000, 500, step=100, help="Number of training iterations for TimeGAN.")
n_samples = st.sidebar.slider("Number of Sample Sequences", 20, 500, 100, step=20, help="Number of Real sequences to fetch and Synthetic sequences to generate.")

if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'real_data' not in st.session_state:
    st.session_state.real_data = None

if st.sidebar.button("Train Vector & Generate Data"):
    
    st.info(f"Extracting Real Data for Scenario: {scenario}")
    
    down_progress = st.progress(0, text="Evaluating Scenario Data...")
    
    def on_data_progress(text, ratio):
        val = max(0, min(100, ratio))
        down_progress.progress(val, text=text)
        
    try:
        real_data = data_processor.load_scenario_data(scenario=scenario, seq_len=24, n_samples=n_samples, progress_callback=on_data_progress)
        st.session_state.real_data = real_data
        down_progress.progress(100, text="Data fetch complete!")
    except Exception as e:
        st.error(f"Error extracting data: {str(e)}")
        st.stop()
        
    st.write("### Initiating TimeGAN Architecture (4-Networks)")
    
    tgan_progress = st.progress(0, text="Starting Training...")
    
    def on_tgan_progress(phase, current_it, total_it):
        ratio = int((current_it / total_it) * 100)
        tgan_progress.progress(ratio, text=f"[{phase}] Step {current_it}/{total_it}")
        
    parameters = {
        'hidden_dim': 24,
        'num_layers': 3,
        'iterations': iterations,
        'batch_size': min(32, n_samples),
        'z_dim': 1
    }
    
    try:
        with st.spinner("Training GAN..."):
            synthetic_data = tgan(real_data, parameters, progress_callback=on_tgan_progress)
        st.session_state.synthetic_data = synthetic_data
        tgan_progress.progress(100, text="Training Complete!")
        st.success("Successfully Synthesized Data preserving original physical correlations.")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        
if st.session_state.synthetic_data is not None and st.session_state.real_data is not None:
    synthetic = st.session_state.synthetic_data
    real = st.session_state.real_data
    
    st.markdown("---")
    st.subheader(f"Temporal Distribution Physics Check: {scenario}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Real Data
    axes[0].set_title("Real Data Sequences")
    axes[0].set_xlabel("Time Step (Hour)")
    axes[0].set_ylabel("Load")
    for r_seq in real:
        axes[0].plot(r_seq, color='blue', alpha=0.3)
        
    # Plot Synthetic Data
    axes[1].set_title("Synthetic Data Sequences (TimeGAN)")
    axes[1].set_xlabel("Time Step (Hour)")
    for s_seq in synthetic:
        axes[1].plot(s_seq, color='red', alpha=0.3)
        
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Energy Turing Test")
    st.write("Can you distinguish between a fundamental Real sequence and a generator's Synthetic sequence? One is selected randomly from each distribution.")
    
    # Randomly select one sequence from each
    turing_real_idx = np.random.randint(0, len(real))
    turing_syn_idx = np.random.randint(0, len(synthetic))
    
    is_real_first = np.random.rand() > 0.5
    
    fig_tt, ax_tt = plt.subplots(figsize=(10, 4))
    
    if is_real_first:
        ax_tt.plot(real[turing_real_idx], label="Sequence A", color="#1f77b4", linewidth=2)
        ax_tt.plot(synthetic[turing_syn_idx], label="Sequence B", color="#ff7f0e", linewidth=2, linestyle='--')
        answer = "🎯 **Sequence A** is Real, and **Sequence B** is Synthetic!"
    else:
        ax_tt.plot(synthetic[turing_syn_idx], label="Sequence A", color="#1f77b4", linewidth=2)
        ax_tt.plot(real[turing_real_idx], label="Sequence B", color="#ff7f0e", linewidth=2, linestyle='--')
        answer = "🎯 **Sequence A** is Synthetic, and **Sequence B** is Real!"

    ax_tt.set_xlabel("Time Step (Hour)")
    ax_tt.set_ylabel("Load")
    ax_tt.set_title("Which sequence is Real and which is Synthetic?")
    ax_tt.legend()
    st.pyplot(fig_tt)
    
    st.button("🔄 Generate New Turing Test Pair")
    
    with st.expander("Reveal Answer 🤔"):
        st.markdown(answer)
    
    st.markdown("---")
    st.subheader("Data Export")
    
    # Flatten data for CSV
    syn_flat = [seq.flatten() for seq in synthetic]
    df_synthetic = pd.DataFrame(syn_flat)
    df_synthetic.index.name = "Sequence_ID"
    df_synthetic.columns = [f"Hour_{i}" for i in range(df_synthetic.shape[1])]
    
    csv = df_synthetic.to_csv().encode('utf-8')
    st.download_button(
        label="Download Synthetic Data as CSV",
        data=csv,
        file_name=f'synthetic_{scenario.replace(" ", "_").lower()}.csv',
        mime='text/csv',
    )
