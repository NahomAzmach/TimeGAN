"""
Evaluation metrics from the TimeGAN paper (Section 5):
  - Discriminative score: post-hoc RNN classifier trained to tell real from synthetic.
    Reported as |accuracy - 0.5|, so 0 is ideal (indistinguishable) and 0.5 is worst.
  - Predictive score ("train on synthetic, test on real"): post-hoc RNN trained on
    synthetic data for one-step-ahead prediction, evaluated by MAE on real data.
    Compared against the same model trained on real data (TRTR) as a reference.
  - PCA / t-SNE embeddings of real vs synthetic sequences for a visual diversity check.

Unlike the paper's single 80/20 split, scores here use k-fold cross-validation so every
sequence is scored once. With only ~100 sequences per class, a 20% hold-out is ~20 samples,
which made the discriminative score swing by +-0.1 between runs.

The post-hoc models are also larger than the paper's (hidden_dim = data_dim/2). For univariate
data that rule gives a 4-unit GRU that is still improving at 3000+ iterations, so the score
depended mostly on the training budget. A 24-unit GRU plateaus by ~1000 iterations.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def _normalize_like(dataX, ref):
    # Scale both datasets with the *real* data's min/max so errors are comparable
    stacked = np.vstack(ref)
    min_val, max_val = np.min(stacked, axis=0), np.max(stacked, axis=0)
    return np.array([(x - min_val) / (max_val - min_val + 1e-7) for x in dataX])

def _kfold(n, k):
    """Returns a list of k (train_idx, test_idx) pairs covering all n indices once."""
    folds = np.array_split(np.random.permutation(n), k)
    return [(np.concatenate(folds[:i] + folds[i+1:]), folds[i]) for i in range(k)]

def chance_level(n_real, n_synthetic):
    """
    Expected discriminative score for *perfect* synthetic data, purely from finite test size:
    accuracy ~ N(0.5, 0.25/N), so E|accuracy - 0.5| = sqrt(2/pi) * sqrt(0.25/N).
    Scores near this are indistinguishable from real at this sample size.
    """
    return float(np.sqrt(2 / np.pi) * np.sqrt(0.25 / (n_real + n_synthetic)))

def discriminative_score(real, synthetic, iterations=1000, batch_size=64, hidden_dim=24, k=5):
    """
    real, synthetic: lists of [Seq_Len, Data_Dim] arrays (same length/dim).
    Returns |cross-validated accuracy - 0.5|, where every real and synthetic sequence
    is classified exactly once by a classifier that didn't train on it.
    """
    tf.reset_default_graph()
    tf.set_random_seed(np.random.randint(2**31))  # follows the numpy seed for reproducibility

    real_n = _normalize_like(real, real)
    syn_n = _normalize_like(synthetic, real)
    seq_len, dim = real_n.shape[1], real_n.shape[2]

    X = tf.placeholder(tf.float32, [None, seq_len, dim], name="disc_x")
    X_hat = tf.placeholder(tf.float32, [None, seq_len, dim], name="disc_x_hat")

    def classifier(x):
        with tf.variable_scope("post_hoc_discriminator", reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
            _, last_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            logit = tf.layers.dense(last_state, 1, activation=None)
        return logit

    logit_real = classifier(X)
    logit_fake = classifier(X_hat)
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('post_hoc_discriminator')]

    d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(logit_real), logit_real) + \
             tf.losses.sigmoid_cross_entropy(tf.zeros_like(logit_fake), logit_fake)
    d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
    init = tf.global_variables_initializer()

    # Folds are split per class so each fold keeps the real/synthetic balance
    real_folds, syn_folds = _kfold(len(real_n), k), _kfold(len(syn_n), k)
    correct = 0

    with tf.Session() as sess:
        for (real_train, real_test), (syn_train, syn_test) in zip(real_folds, syn_folds):
            sess.run(init)  # fresh classifier per fold; reuses the graph instead of rebuilding it
            for _ in range(iterations):
                r_mb = real_n[np.random.choice(real_train, min(batch_size, len(real_train)), replace=False)]
                s_mb = syn_n[np.random.choice(syn_train, min(batch_size, len(syn_train)), replace=False)]
                sess.run(d_solver, feed_dict={X: r_mb, X_hat: s_mb})

            l_real, l_fake = sess.run([logit_real, logit_fake],
                                      feed_dict={X: real_n[real_test], X_hat: syn_n[syn_test]})
            correct += np.sum(l_real > 0) + np.sum(l_fake <= 0)

    accuracy = correct / (len(real_n) + len(syn_n))
    return abs(accuracy - 0.5)

class _OneStepPredictor:
    """GRU that predicts x[t+1] from x[:t+1]. One graph, re-initialized per fit."""

    def __init__(self, seq_len, dim, hidden_dim):
        tf.reset_default_graph()
        tf.set_random_seed(np.random.randint(2**31))

        self.X = tf.placeholder(tf.float32, [None, seq_len - 1, dim], name="pred_x")
        self.Y = tf.placeholder(tf.float32, [None, seq_len - 1, dim], name="pred_y")

        with tf.variable_scope("post_hoc_predictor"):
            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
            outputs, _ = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
            self.Y_pred = tf.layers.dense(outputs, dim, activation=tf.nn.sigmoid)

        p_loss = tf.losses.absolute_difference(self.Y, self.Y_pred)
        self.solver = tf.train.AdamOptimizer().minimize(p_loss)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()

    def fit_score(self, train_data, test_data, iterations, batch_size):
        """Trains from scratch on train_data; returns per-sequence MAE on test_data."""
        self.sess.run(self.init)
        for _ in range(iterations):
            mb = train_data[np.random.choice(len(train_data), min(batch_size, len(train_data)), replace=False)]
            self.sess.run(self.solver, feed_dict={self.X: mb[:, :-1, :], self.Y: mb[:, 1:, :]})
        pred = self.sess.run(self.Y_pred, feed_dict={self.X: test_data[:, :-1, :]})
        return np.abs(pred - test_data[:, 1:, :]).mean(axis=(1, 2))

    def close(self):
        self.sess.close()

def predictive_score(real, synthetic, iterations=1000, batch_size=64, hidden_dim=24, k=5):
    """
    Returns (tstr_mae, trtr_mae), both averaged over *every* real sequence:
      tstr_mae: Train on (all) Synthetic, Test on Real.
      trtr_mae: Train on Real, Test on Real via k-fold, so each real day is predicted by a
                model that never saw it - the reference to beat.
    A TSTR close to TRTR means the synthetic data is about as useful as the real data.
    """
    real_n = _normalize_like(real, real)
    syn_n = _normalize_like(synthetic, real)

    model = _OneStepPredictor(real_n.shape[1], real_n.shape[2], hidden_dim)
    try:
        tstr = model.fit_score(syn_n, real_n, iterations, batch_size).mean()
        trtr_errors = np.zeros(len(real_n))
        for train_idx, test_idx in _kfold(len(real_n), k):
            trtr_errors[test_idx] = model.fit_score(real_n[train_idx], real_n[test_idx], iterations, batch_size)
    finally:
        model.close()
    return float(tstr), float(trtr_errors.mean())

def evaluate(real, synthetic, n_runs=3, progress_callback=None):
    """
    Runs both scores n_runs times and returns mean/std plus the chance level for this sample size.
    """
    disc, tstr, trtr = [], [], []
    for run in range(n_runs):
        if progress_callback: progress_callback(f"Discriminative score (run {run + 1}/{n_runs})", run * 2, n_runs * 2)
        disc.append(discriminative_score(real, synthetic))
        if progress_callback: progress_callback(f"Predictive score (run {run + 1}/{n_runs})", run * 2 + 1, n_runs * 2)
        t, r = predictive_score(real, synthetic)
        tstr.append(t)
        trtr.append(r)
    if progress_callback: progress_callback("Done", n_runs * 2, n_runs * 2)

    return {
        'discriminative': (float(np.mean(disc)), float(np.std(disc))),
        'tstr_mae': (float(np.mean(tstr)), float(np.std(tstr))),
        'trtr_mae': (float(np.mean(trtr)), float(np.std(trtr))),
        'chance_level': chance_level(len(real), len(synthetic)),
    }

def embed_2d(real, synthetic, method="pca", max_points=500):
    """
    Projects each sequence to 2D, as in the paper (Figure 3): sequences are averaged
    over the feature axis, giving one Seq_Len-dimensional vector per sample.
    Subsamples to max_points per class so t-SNE stays fast on pooled datasets.
    Returns (real_2d, synthetic_2d).
    """
    real_n = _normalize_like(real, real).mean(axis=2)
    syn_n = _normalize_like(synthetic, real).mean(axis=2)
    real_n = real_n[np.random.permutation(len(real_n))[:max_points]]
    syn_n = syn_n[np.random.permutation(len(syn_n))[:max_points]]

    if method == "pca":
        pca = PCA(n_components=2).fit(real_n)
        return pca.transform(real_n), pca.transform(syn_n)

    combined = np.concatenate([real_n, syn_n], axis=0)
    perplexity = min(40, len(combined) - 1)
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(combined)
    return emb[:len(real_n)], emb[len(real_n):]

def condition_adherence(generate_fn, conditions, feature_names, shape_features_fn):
    """
    Does a conditional generator actually follow its condition?
    generate_fn: conditions [N, k] -> sequences [N, Seq_Len, 1]
    conditions: [N, k] raw condition vectors (ideally from held-out real days)
    feature_names: names of the k condition columns; the first m are measurable from a curve
    shape_features_fn: sequences -> [N, m] measured features (same order as the first m columns)

    For each measurable feature returns the mean absolute error between requested and measured,
    for (a) the real conditions and (b) the same conditions shuffled across days. If (a) isn't
    clearly below (b), the generator is ignoring that part of the condition.
    """
    conditions = np.asarray(conditions)
    measured = shape_features_fn(generate_fn(conditions))
    shuffled = conditions[np.random.permutation(len(conditions))]
    measured_shuffled = shape_features_fn(generate_fn(shuffled))

    results = {}
    for j in range(measured.shape[1]):
        requested = conditions[:, j]
        results[feature_names[j]] = {
            'conditioned_mae': float(np.mean(np.abs(measured[:, j] - requested))),
            'shuffled_mae': float(np.mean(np.abs(measured_shuffled[:, j] - requested))),
        }
    # Peak hour is the one users will check by eye, so also report "within 1 hour"
    if 'peak_hour' in feature_names:
        j = feature_names.index('peak_hour')
        results['peak_hour']['within_1h'] = float(np.mean(np.abs(measured[:, j] - conditions[:, j]) <= 1))
        results['peak_hour']['within_1h_shuffled'] = float(np.mean(np.abs(measured_shuffled[:, j] - conditions[:, j]) <= 1))
    return results
