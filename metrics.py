"""
Evaluation metrics from the TimeGAN paper (Section 5):
  - Discriminative score: post-hoc RNN classifier trained to tell real from synthetic.
    Reported as |accuracy - 0.5|, so 0 is ideal (indistinguishable) and 0.5 is worst.
  - Predictive score ("train on synthetic, test on real"): post-hoc RNN trained on
    synthetic data for one-step-ahead prediction, evaluated by MAE on held-out real data.
    Compared against the same model trained on real data (TRTR) as a reference.
  - PCA / t-SNE embeddings of real vs synthetic sequences for a visual diversity check.
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

def _train_test_split(n, train_rate=0.8):
    idx = np.random.permutation(n)
    cut = int(n * train_rate)
    return idx[:cut], idx[cut:]

def discriminative_score(real, synthetic, iterations=1000, batch_size=64, hidden_dim=None):
    """
    real, synthetic: lists of [Seq_Len, Data_Dim] arrays (same length/dim).
    Returns |test accuracy - 0.5|.
    """
    tf.reset_default_graph()
    tf.set_random_seed(np.random.randint(2**31))  # follows the numpy seed for reproducibility

    real_n = _normalize_like(real, real)
    syn_n = _normalize_like(synthetic, real)
    seq_len, dim = real_n.shape[1], real_n.shape[2]
    hidden_dim = hidden_dim or max(dim // 2, 4)

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

    real_train, real_test = _train_test_split(len(real_n))
    syn_train, syn_test = _train_test_split(len(syn_n))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(iterations):
            r_mb = real_n[np.random.choice(real_train, min(batch_size, len(real_train)), replace=False)]
            s_mb = syn_n[np.random.choice(syn_train, min(batch_size, len(syn_train)), replace=False)]
            sess.run(d_solver, feed_dict={X: r_mb, X_hat: s_mb})

        p_real, p_fake = sess.run([tf.sigmoid(logit_real), tf.sigmoid(logit_fake)],
                                  feed_dict={X: real_n[real_test], X_hat: syn_n[syn_test]})

    correct = np.sum(p_real > 0.5) + np.sum(p_fake <= 0.5)
    accuracy = correct / (len(p_real) + len(p_fake))
    return abs(accuracy - 0.5)

def _one_step_mae(train_data, test_data, iterations, batch_size, hidden_dim):
    """Train a GRU to predict x[t+1] from x[:t+1] on train_data; return MAE on test_data."""
    tf.reset_default_graph()
    tf.set_random_seed(np.random.randint(2**31))  # follows the numpy seed for reproducibility

    seq_len, dim = train_data.shape[1], train_data.shape[2]
    X = tf.placeholder(tf.float32, [None, seq_len - 1, dim], name="pred_x")
    Y = tf.placeholder(tf.float32, [None, seq_len - 1, dim], name="pred_y")

    with tf.variable_scope("post_hoc_predictor"):
        cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
        outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        Y_pred = tf.layers.dense(outputs, dim, activation=tf.nn.sigmoid)

    p_loss = tf.losses.absolute_difference(Y, Y_pred)
    p_solver = tf.train.AdamOptimizer().minimize(p_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(iterations):
            mb = train_data[np.random.choice(len(train_data), min(batch_size, len(train_data)), replace=False)]
            sess.run(p_solver, feed_dict={X: mb[:, :-1, :], Y: mb[:, 1:, :]})

        pred = sess.run(Y_pred, feed_dict={X: test_data[:, :-1, :]})

    return float(np.mean(np.abs(pred - test_data[:, 1:, :])))

def predictive_score(real, synthetic, iterations=1000, batch_size=64, hidden_dim=None):
    """
    Returns (tstr_mae, trtr_mae):
      tstr_mae: Train on Synthetic, Test on held-out Real.
      trtr_mae: Train on Real (80%), Test on the same held-out Real (20%) - the reference to beat.
    A TSTR close to TRTR means the synthetic data is about as useful as the real data.
    """
    real_n = _normalize_like(real, real)
    syn_n = _normalize_like(synthetic, real)
    hidden_dim = hidden_dim or max(real_n.shape[2] // 2, 4)

    real_train, real_test = _train_test_split(len(real_n))
    tstr = _one_step_mae(syn_n, real_n[real_test], iterations, batch_size, hidden_dim)
    trtr = _one_step_mae(real_n[real_train], real_n[real_test], iterations, batch_size, hidden_dim)
    return tstr, trtr

def evaluate(real, synthetic, n_runs=3, iterations=1000, progress_callback=None):
    """
    Runs both scores n_runs times (they're noisy on small datasets) and returns mean/std.
    """
    disc, tstr, trtr = [], [], []
    for run in range(n_runs):
        if progress_callback: progress_callback(f"Discriminative score (run {run + 1}/{n_runs})", run * 2, n_runs * 2)
        disc.append(discriminative_score(real, synthetic, iterations=iterations))
        if progress_callback: progress_callback(f"Predictive score (run {run + 1}/{n_runs})", run * 2 + 1, n_runs * 2)
        t, r = predictive_score(real, synthetic, iterations=iterations)
        tstr.append(t)
        trtr.append(r)
    if progress_callback: progress_callback("Done", n_runs * 2, n_runs * 2)

    return {
        'discriminative': (float(np.mean(disc)), float(np.std(disc))),
        'tstr_mae': (float(np.mean(tstr)), float(np.std(tstr))),
        'trtr_mae': (float(np.mean(trtr)), float(np.std(trtr))),
    }

def embed_2d(real, synthetic, method="pca"):
    """
    Projects each sequence to 2D, as in the paper (Figure 3): sequences are averaged
    over the feature axis, giving one Seq_Len-dimensional vector per sample.
    Returns (real_2d, synthetic_2d).
    """
    real_n = _normalize_like(real, real).mean(axis=2)
    syn_n = _normalize_like(synthetic, real).mean(axis=2)

    if method == "pca":
        pca = PCA(n_components=2).fit(real_n)
        return pca.transform(real_n), pca.transform(syn_n)

    combined = np.concatenate([real_n, syn_n], axis=0)
    perplexity = min(40, len(combined) - 1)
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(combined)
    return emb[:len(real_n)], emb[len(real_n):]
