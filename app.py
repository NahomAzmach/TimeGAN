import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from timegan_core import tgan
import data_processor
import metrics

st.set_page_config(page_title="TimeGAN: Synthetic Data Generator", page_icon="📈", layout="wide")

st.title("TimeGAN: Synthetic Energy Data Generator")
st.markdown("""
This application implements the  **Time-series Generative Adversarial Networks (TimeGAN)**.
It generates realistic time-series data while preserving the temporal dynamics (e.g. Load Surge) inherent in the training data using a **Stepwise Supervised Loss** via the Supervisor network.
""")

@st.cache_data(show_spinner="Fetching BDG-2 building list...")
def get_buildings():
    try:
        return data_processor.list_buildings(), data_processor.list_building_options()
    except Exception:
        return [data_processor.DEFAULT_BUILDING], [data_processor.DEFAULT_BUILDING]

st.sidebar.header("Configuration")
buildings, building_options = get_buildings()
building = st.sidebar.selectbox("Building (BDG-2 Panther, 2017)", building_options,
                                index=building_options.index(data_processor.DEFAULT_BUILDING) if data_processor.DEFAULT_BUILDING in building_options else 0,
                                format_func=lambda b: data_processor.describe_option(b, buildings),
                                help="Real hourly load from the Building Data Genome 2 dataset. Each day becomes one 24-hour sequence. "
                                     "Pooled groups combine every building of that type (each scaled by its own average load), giving far more days to train and evaluate on.")
scenario = st.sidebar.selectbox("Scenario Description", ["Baseline", "Load Surge"], help="Select the real-world temporal pattern you want to synthesize.")

iterations = st.sidebar.slider("Training Iterations", 100, 5000, 500, step=100, help="Number of training iterations for TimeGAN.")
n_samples = st.sidebar.slider("Number of Sample Sequences", 20, 1000, 100, step=20, help="Number of Real sequences to fetch and Synthetic sequences to generate. A single building-year has ~365 days, so real data is capped at half of that per scenario; pooled groups allow the full range.")
seed = st.sidebar.number_input("Random Seed", value=42, step=1, help="Fixes numpy/TensorFlow randomness so runs are reproducible.")

if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'real_data' not in st.session_state:
    st.session_state.real_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

if st.sidebar.button("Train Vector & Generate Data"):
    np.random.seed(int(seed))
    st.session_state.metrics = None

    st.info(f"Extracting Real Data for Scenario: {scenario}")

    down_progress = st.progress(0, text="Evaluating Scenario Data...")

    def on_data_progress(text, ratio):
        val = max(0, min(100, ratio))
        down_progress.progress(val, text=text)

    try:
        real_data, source = data_processor.load_scenario_data(scenario=scenario, seq_len=24, n_samples=n_samples, building=building, progress_callback=on_data_progress)
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
        'batch_size': min(32, len(real_data)),
        'z_dim': 1,
        'seed': int(seed)
    }

    try:
        with st.spinner("Training GAN..."):
            synthetic_data = tgan(real_data, parameters, progress_callback=on_tgan_progress)
        st.session_state.synthetic_data = synthetic_data
        # Remember what this model was trained on, independent of later sidebar changes
        st.session_state.trained_on = {'scenario': scenario, 'building': building, 'source': source, 'iterations': iterations}
        tgan_progress.progress(100, text="Training Complete!")
        st.success("Successfully Synthesized Data preserving original physical correlations.")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

if st.session_state.synthetic_data is not None and st.session_state.real_data is not None:
    synthetic = st.session_state.synthetic_data
    real = st.session_state.real_data
    trained_on = st.session_state.trained_on

    st.markdown("---")
    st.subheader(f"Temporal Distribution Physics Check: {trained_on['scenario']}")
    if trained_on['source'] == "mock fallback":
        st.warning("Real data couldn't be downloaded, so the 'Real' sequences below come from the mathematical fallback generator.")
    else:
        st.caption(f"Real data: {len(real)} days from **{data_processor.describe_option(trained_on['source'], buildings)}** (BDG-2 Panther, 2017). Model trained for {trained_on['iterations']} iterations per phase.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    # Plot Real Data
    axes[0].set_title("Real Data Sequences")
    axes[0].set_xlabel("Time Step (Hour)")
    axes[0].set_ylabel("Load (× building average)" if data_processor.is_group(trained_on['source']) else "Load")
    for r_seq in real:
        axes[0].plot(r_seq, color='blue', alpha=0.3)

    # Plot Synthetic Data
    axes[1].set_title("Synthetic Data Sequences (TimeGAN)")
    axes[1].set_xlabel("Time Step (Hour)")
    for s_seq in synthetic:
        axes[1].plot(s_seq, color='red', alpha=0.3)

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Quantitative Evaluation")
    st.write("The paper's post-hoc metrics. Each trains a small GRU with 5-fold cross-validation, so every sequence is scored once by a model that never saw it.")

    eval_runs = st.slider("Evaluation runs", 1, 10, 2, help="More runs give a tighter mean ± std but take longer (~35s per run).")

    if st.button("Compute Metrics"):
        eval_progress = st.progress(0, text="Starting evaluation...")

        def on_eval_progress(text, current, total):
            eval_progress.progress(int(current / total * 100), text=text)

        np.random.seed(int(seed))
        scores = metrics.evaluate(real, synthetic, n_runs=eval_runs, progress_callback=on_eval_progress)
        with st.spinner("Computing PCA and t-SNE embeddings..."):
            scores['pca'] = metrics.embed_2d(real, synthetic, "pca")
            scores['tsne'] = metrics.embed_2d(real, synthetic, "tsne")
        st.session_state.metrics = scores
        eval_progress.progress(100, text="Evaluation complete!")

    if st.session_state.metrics is not None:
        scores = st.session_state.metrics
        d_mean, d_std = scores['discriminative']
        tstr_mean, tstr_std = scores['tstr_mae']
        trtr_mean, trtr_std = scores['trtr_mae']

        col1, col2, col3 = st.columns(3)
        col1.metric("Discriminative Score ↓", f"{d_mean:.3f} ± {d_std:.3f}",
                    help="|accuracy − 0.5| of a GRU classifier trained to separate real from synthetic. 0.5 = trivially separable. "
                         "Even perfect synthetic data scores around the chance level, because a finite test set never gives exactly 50% accuracy.")
        col2.metric("Predictive MAE (TSTR) ↓", f"{tstr_mean:.4f} ± {tstr_std:.4f}",
                    delta=f"{tstr_mean - trtr_mean:+.4f} vs real", delta_color="inverse",
                    help="Train on Synthetic, Test on Real: a GRU learns one-step-ahead prediction from synthetic data and is scored on held-out real days (normalized units).")
        col3.metric("Reference MAE (TRTR)", f"{trtr_mean:.4f} ± {trtr_std:.4f}",
                    help="Same predictor trained on 80% of the real days, tested on the other 20%. TSTR close to this means the synthetic data is as useful as the real data.")

        st.caption(f"Chance level for {len(real)} real + {len(synthetic)} synthetic sequences: a discriminative score of ≈ {scores['chance_level']:.3f} "
                   "means indistinguishable from real. Compare configurations only when their ± ranges don't overlap.")
        
        fig_emb, axes_emb = plt.subplots(1, 2, figsize=(15, 5))
        for ax, key, title in [(axes_emb[0], 'pca', "PCA"), (axes_emb[1], 'tsne', "t-SNE")]:
            real_2d, syn_2d = scores[key]
            ax.scatter(real_2d[:, 0], real_2d[:, 1], color='blue', alpha=0.4, label="Real")
            ax.scatter(syn_2d[:, 0], syn_2d[:, 1], color='red', alpha=0.4, label="Synthetic")
            ax.set_title(f"{title}: Real vs Synthetic")
            ax.legend()
        st.pyplot(fig_emb)
        st.caption("Good synthetic data overlaps the real points and covers the same spread. A tight red cluster inside a wider blue cloud signals mode collapse.")

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
        file_name=f'synthetic_{trained_on["scenario"].replace(" ", "_").lower()}.csv',
        mime='text/csv',
    )
