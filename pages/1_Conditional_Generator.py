import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_processor
import metrics
import plots
import timegan_torch

st.set_page_config(page_title="TimeGAN: Conditional Generator", page_icon="🎛️", layout="wide")

st.title("Conditional Generator")
st.markdown("""
One TimeGAN trained on **every** day, where each day is described by a condition vector
(average load, peak hour, surge strength, weekend, temperature). Instead of training a separate model per scenario,
you describe the day you want and the generator produces it. This is the input a plain-text scenario feature would fill in.
""")

FEATURES = data_processor.CONDITION_FEATURES
LABELS = data_processor.FEATURE_LABELS

@st.cache_data(show_spinner="Fetching BDG-2 building list...")
def get_buildings():
    try:
        return data_processor.list_buildings(), data_processor.list_building_options()
    except Exception:
        return [data_processor.DEFAULT_BUILDING], [data_processor.DEFAULT_BUILDING]

@st.cache_data(show_spinner="Building daily condition vectors...")
def get_conditional_data(building):
    return data_processor.load_conditional_data(building)

st.sidebar.header("Training")
buildings, building_options = get_buildings()
default_option = f"{data_processor.GROUP_PREFIX}office"
building = st.sidebar.selectbox("Building (BDG-2 Panther, 2017)", building_options,
                                index=building_options.index(default_option) if default_option in building_options else 0,
                                format_func=lambda b: data_processor.describe_option(b, buildings),
                                help="A pooled group gives thousands of days, which this model needs; a single building has ~365.")
iterations = st.sidebar.slider("Training Iterations", 500, 5000, 2000, step=500,
                               help="Per phase. ~70s for 2000 on an RTX 5050 with the office group.")
condition_weight = st.sidebar.slider("Condition loss weight", 0, 100, 50, step=10,
                                     help="Weight of the auxiliary loss that measures mean, peak hour and surge strength on generated days "
                                          "and penalizes mismatches. At 0, only the discriminator enforces the condition, and in testing "
                                          "the model then ignored surge strength completely.")
z_dim = st.sidebar.select_slider("Noise dimensions (z_dim)", options=[1, 2, 4, 8, 16], value=1,
                                 help="Size of the generator's random input. With 1, days generated for the same condition come out "
                                      "nearly identical (~11% of the real variety in testing); 8 gave ~46% of the real variety and "
                                      "realistic hour-to-hour noise, but followed the surge-strength condition less closely.")
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

if st.sidebar.button("Train Conditional Model"):
    X, C, dates = get_conditional_data(building)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    train_idx, test_idx = idx[:cut], idx[cut:]

    progress = st.progress(0, text="Starting Training...")

    def on_progress(phase, current_it, total_it):
        progress.progress(int(current_it / total_it * 100), text=f"[{phase}] Step {current_it}/{total_it}")

    model = timegan_torch.ConditionalTimeGAN({
        'hidden_dim': 24,
        'num_layers': 3,
        'iterations': iterations,
        'batch_size': 128,
        'z_dim': z_dim,
        'seed': int(seed),
        'condition_loss_weight': condition_weight,
    })
    with st.spinner(f"Training on {len(train_idx)} days..."):
        model.fit(X[train_idx], C[train_idx], progress_callback=on_progress)
    progress.progress(100, text="Training Complete!")

    st.session_state.cond_model = model
    st.session_state.cond_data = {'X': X, 'C': C, 'dates': dates, 'train_idx': train_idx, 'test_idx': test_idx,
                                  'building': building, 'iterations': iterations, 'condition_weight': condition_weight, 'z_dim': z_dim}
    st.session_state.cond_eval = None
    st.session_state.cond_generated = None

if st.session_state.get('cond_model') is None:
    st.info("Train a model from the sidebar to start generating days.")
    st.stop()

model = st.session_state.cond_model
data = st.session_state.cond_data
X, C, dates = data['X'], data['C'], data['dates']
C_train = C[data['train_idx']]
is_group = data_processor.is_group(data['building'])
load_label = "Load (× building average)" if is_group else "Load"

st.caption(f"Model trained on {len(data['train_idx'])} days from **{data_processor.describe_option(data['building'], buildings)}** "
           f"for {data['iterations']} iterations per phase (condition loss weight {data['condition_weight']}, z_dim {data.get('z_dim', 1)}) "
           f"on {timegan_torch.device_name()}. {len(data['test_idx'])} days are held out for evaluation.")

# ---------------------------------------------------------------------------
# Describe the day
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Describe the Day")

def preset_from(mask):
    """Median condition of the real training days matching mask: what such a day typically looks like."""
    rows = C_train[mask]
    values = np.median(rows, axis=0)
    values[FEATURES.index('peak_hour')] = np.round(values[FEATURES.index('peak_hour')])
    values[FEATURES.index('weekend')] = np.round(values[FEATURES.index('weekend')])
    return values

col_ix = {f: i for i, f in enumerate(FEATURES)}
weekday = C_train[:, col_ix['weekend']] == 0
presets = {
    "Typical weekday": preset_from(weekday),
    "Load surge": preset_from(weekday & (C_train[:, col_ix['peak_ratio']] >= np.percentile(C_train[:, col_ix['peak_ratio']], 95))),
    "Quiet weekend": preset_from(~weekday),
    "Hot weekday": preset_from(weekday & (C_train[:, col_ix['temperature']] >= np.percentile(C_train[:, col_ix['temperature']], 90))),
}

lo = np.percentile(C_train, 1, axis=0)
hi = np.percentile(C_train, 99, axis=0)

def set_condition(values):
    for f, v in zip(FEATURES, values):
        if f == 'weekend':
            st.session_state[f"cond_{f}"] = bool(v)
        elif f == 'peak_hour':
            st.session_state[f"cond_{f}"] = int(v)
        else:
            st.session_state[f"cond_{f}"] = float(np.clip(v, lo[col_ix[f]], hi[col_ix[f]]))

if "cond_mean_load" not in st.session_state:
    set_condition(presets["Typical weekday"])

preset_cols = st.columns(len(presets))
for col, (name, values) in zip(preset_cols, presets.items()):
    col.button(name, on_click=set_condition, args=(values,), use_container_width=True,
               help="Sets the sliders to the median condition of real days of this kind.")

c1, c2, c3 = st.columns(3)
c1.slider(LABELS['mean_load'], float(lo[0]), float(hi[0]), key="cond_mean_load", step=0.01,
          help="Daily average load" + (" relative to each building's yearly average (1.0 = an average day)." if is_group else "."))
c1.slider(LABELS['peak_hour'], 0, 23, key="cond_peak_hour", help="Hour of the day's maximum load.")
c2.slider(LABELS['peak_ratio'], float(lo[2]), float(hi[2]), key="cond_peak_ratio", step=0.01,
          help="Daily maximum divided by daily average: 1.0 is flat, 2.0 is a surge to twice the average.")
c2.slider(LABELS['temperature'], float(lo[4]), float(hi[4]), key="cond_temperature", step=0.1,
          help="Daily mean outdoor temperature at the Panther site.")
c3.checkbox(LABELS['weekend'], key="cond_weekend")
n_generate = c3.slider("Days to generate", 5, 100, 20, step=5)

requested = np.array([st.session_state.cond_mean_load, st.session_state.cond_peak_hour, st.session_state.cond_peak_ratio,
                      float(st.session_state.cond_weekend), st.session_state.cond_temperature])

if st.button("Generate Days", type="primary"):
    st.session_state.cond_generated = (requested, model.generate(np.tile(requested, (n_generate, 1))))

if st.session_state.get('cond_generated') is not None:
    gen_requested, generated = st.session_state.cond_generated

    # Real days with the most similar conditions (distance in the model's [0, 1] condition scale)
    scale = model.cond_max - model.cond_min + 1e-7
    distance = np.abs((C - gen_requested) / scale).sum(axis=1)
    nearest = np.argsort(distance)[:len(generated)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for seq in generated:
        axes[0].plot(seq, color='red', alpha=0.4)
    axes[0].set_title(f"Generated ({len(generated)} days)")
    axes[0].set_ylabel(load_label)
    for i in nearest:
        axes[1].plot(X[i], color='blue', alpha=0.4)
    axes[1].set_title(f"Real days with the most similar conditions ({len(nearest)})")
    for ax in axes:
        ax.set_xlabel("Time Step (Hour)")
        ax.axvline(gen_requested[col_ix['peak_hour']], color='gray', linestyle=':')
    st.pyplot(fig)

    measured = data_processor.shape_features(generated)
    measured_real = data_processor.shape_features(X[nearest])
    table = pd.DataFrame({
        "Requested": gen_requested[:3],
        "Generated (mean)": measured.mean(axis=0),
        "Similar real days (mean)": measured_real.mean(axis=0),
    }, index=[LABELS[f] for f in data_processor.SHAPE_FEATURES])
    st.dataframe(table.style.format("{:.2f}"))
    spread_gen = generated[:, :, 0].std(axis=0).mean()
    spread_real = X[nearest][:, :, 0].std(axis=0).mean()
    st.caption("The dotted line marks the requested peak hour. Weekend and temperature can't be measured from a single curve, "
               "so they're compared through the similar real days instead. "
               f"Variety between the generated days is {spread_gen / spread_real:.0%} of the variety between the similar real days.")

    csv_df = pd.DataFrame(generated[:, :, 0], columns=[f"Hour_{h}" for h in range(generated.shape[1])])
    for f, v in zip(FEATURES, gen_requested):
        csv_df[f"condition_{f}"] = v
    st.download_button("Download Generated Days as CSV", csv_df.to_csv(index_label="Sequence_ID").encode('utf-8'),
                       file_name="synthetic_conditional.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Does the Model Follow Its Conditions?")
st.write("Evaluated on the held-out days: the model generates one day for each real day's condition vector, "
         "then the same features are measured on the result. The **shuffled** column gives each held-out day "
         "someone else's condition, so it's the error you'd get from a model that ignores conditions.")

if st.button("Evaluate on Held-out Days"):
    test_idx = data['test_idx']
    with st.spinner("Measuring condition adherence..."):
        np.random.seed(0)
        adherence = metrics.condition_adherence(model.generate, C[test_idx], FEATURES, data_processor.shape_features)
    eval_progress = st.progress(0, text="Scoring realism...")

    def on_eval_progress(text, current, total):
        eval_progress.progress(int(current / total * 100), text=text)

    # Realism: up to 600 held-out days vs one generated day per their conditions
    sub = test_idx[:600]
    np.random.seed(1)
    realism = metrics.evaluate(list(X[sub]), list(model.generate(C[sub])), n_runs=2, progress_callback=on_eval_progress)
    eval_progress.progress(100, text="Evaluation complete!")
    st.session_state.cond_eval = (adherence, realism, len(sub))

if st.session_state.get('cond_eval') is not None:
    adherence, realism, n_eval = st.session_state.cond_eval
    rows = []
    for f in data_processor.SHAPE_FEATURES:
        rows.append({"Feature": LABELS[f], "Error (conditioned)": adherence[f]['conditioned_mae'],
                     "Error (shuffled)": adherence[f]['shuffled_mae']})
    st.dataframe(pd.DataFrame(rows).set_index("Feature").style.format("{:.3f}"))
    st.write(f"Peak hour within ±1 hour of the request: **{adherence['peak_hour']['within_1h']:.0%}** "
             f"(shuffled: {adherence['peak_hour']['within_1h_shuffled']:.0%}).")

    d_mean, d_std = realism['discriminative']
    col1, col2, col3 = st.columns(3)
    col1.metric("Discriminative Score ↓", f"{d_mean:.3f} ± {d_std:.3f}")
    col2.metric("Predictive MAE (TSTR) ↓", f"{realism['tstr_mae'][0]:.4f}",
                delta=f"{realism['tstr_mae'][0] - realism['trtr_mae'][0]:+.4f} vs real", delta_color="inverse")
    col3.metric("Reference MAE (TRTR)", f"{realism['trtr_mae'][0]:.4f}")
    st.caption(f"Realism metrics use {n_eval} held-out real days and one generated day per their conditions. "
               f"Chance level ≈ {realism['chance_level']:.3f}.")

with st.expander("Training Loss Curves"):
    st.pyplot(plots.loss_curves_figure(model.history))
    st.caption(plots.LOSS_CAPTION)
