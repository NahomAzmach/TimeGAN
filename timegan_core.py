import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Ensures TF1 graphs run on modern environments
import numpy as np

def MinMaxScaler(dataX):
    # Flatten the list of arrays for min/max calculation per feature
    # dataX is a list of arrays: each array is [Seq_Len, Data_Dim]
    # We find overall min and max across all sequences
    min_val = np.min(np.vstack(dataX), axis=0)
    dataX_norm = [(x - min_val) / (np.max(np.vstack(dataX), axis=0) - min_val + 1e-7) for x in dataX]
    max_val = np.max(np.vstack(dataX), axis=0)
    return dataX_norm, min_val, max_val

def tgan(dataX, parameters, progress_callback=None):
    """
    TimeGAN Generator using tf.compat.v1
    dataX: List of sequences, e.g. [np.array( [Seq_Len, Data_Dim] ), ...]
    parameters: dict of TimeGAN parameters
    progress_callback: A function(progress_ratio) to update the Streamlit UI
    """
    tf.reset_default_graph()
    # Graph-level seed must be set after the reset, or it's discarded with the old graph
    if parameters.get('seed') is not None:
        tf.set_random_seed(parameters['seed'])

    No = len(dataX)
    data_dim = len(dataX[0][0, :])

    dataT = [len(seq[:, 0]) for seq in dataX]
    Max_Seq_Len = max(dataT)

    dataX_norm, min_val, max_val = MinMaxScaler(dataX)

    hidden_dim = parameters.get('hidden_dim', 24)
    num_layers = parameters.get('num_layers', 3)
    iterations = parameters.get('iterations', 5000) # Recommend low iterations for UI responsiveness
    batch_size = parameters.get('batch_size', 128)
    z_dim = parameters.get('z_dim', data_dim)
    gamma = 1

    # Placeholders
    X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name="myinput_x")
    Z = tf.placeholder(tf.float32, [None, Max_Seq_Len, z_dim], name="myinput_z")
    T = tf.placeholder(tf.int32, [None], name="myinput_t")

    def rnn_cell():
        return tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
        
    def embedder(X, T):
        with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(num_layers)])
            e_outputs, _ = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
            H = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid)
        return H
      
    def recovery(H, T):
        with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
            r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(num_layers)])
            r_outputs, _ = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=T)
            X_tilde = tf.layers.dense(r_outputs, data_dim, activation=tf.nn.sigmoid)
        return X_tilde
    
    def generator(Z, T):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(num_layers)])
            e_outputs, _ = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length=T)
            E = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid)
        return E
    
    def supervisor(H, T):
        with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(num_layers - 1)])
            e_outputs, _ = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
            S = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid)
        return S
      
    def discriminator(H, T):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(num_layers)])
            d_outputs, _ = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length=T)
            Y_hat = tf.layers.dense(d_outputs, 1, activation=None)
        return Y_hat

    def random_generator(batch_size, z_dim, T_mb, Max_Seq_Len):
        Z_mb = []
        for i in range(batch_size):
            Temp = np.zeros([Max_Seq_Len, z_dim])
            Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
            Temp[:T_mb[i], :] = Temp_Z
            Z_mb.append(Temp)
        return Z_mb

    def batch_generator(data, time, indices):
        X_mb = []
        for i in indices:
            seq = data[i]
            # Pad sequence to Max_Seq_Len
            pad_len = Max_Seq_Len - len(seq)
            if pad_len > 0:
                padded_seq = np.pad(seq, ((0, pad_len), (0, 0)), 'constant')
            else:
                padded_seq = seq
            X_mb.append(padded_seq)
        T_mb = [time[i] for i in indices]
        return X_mb, T_mb

    # Network
    H = embedder(X, T)
    X_tilde = recovery(H, T)
    
    E_hat = generator(Z, T)
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)
    
    X_hat = recovery(H_hat, T)
    
    Y_fake = discriminator(H_hat, T)
    Y_real = discriminator(H, T)
    Y_fake_e = discriminator(E_hat, T)
    
    # Variables
    e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
    # Discriminator loss
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
    # Generator loss
    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
    # Stepwise Supervised loss
    G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,1:,:])
    
    # Two Moments Loss (Mean and Variance constraint)
    G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    G_loss_V = G_loss_V1 + G_loss_V2
    
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V 
            
    # Embedder network loss
    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
    E_loss = E_loss0 + 0.1 * G_loss_S
    
    # Optimizers
    E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Embedding Learning
    for itt in range(iterations):
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]
        X_mb, T_mb = batch_generator(dataX_norm, dataT, train_idx)
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
        if progress_callback and itt % 50 == 0:
            progress_callback("Embedding Network Training", itt, iterations)
    
    # Training with Supervised Loss Only
    for itt in range(iterations):
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]
        X_mb, T_mb = batch_generator(dataX_norm, dataT, train_idx)
        Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
        _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
        if progress_callback and itt % 50 == 0:
            progress_callback("Supervised Loss Training", itt, iterations)

    # Joint Training
    for itt in range(iterations):
        # Generator Training (twice)
        for kk in range(2):
            idx = np.random.permutation(No)
            train_idx = idx[:batch_size]
            X_mb, T_mb = batch_generator(dataX_norm, dataT, train_idx)
            Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)

            _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
            _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
           
        # Discriminator Training
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]
        X_mb, T_mb = batch_generator(dataX_norm, dataT, train_idx)
        Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
        
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        if check_d_loss > 0.15:
            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        if progress_callback and itt % 50 == 0:
            progress_callback("Joint Training", itt, iterations)
            
    # Final Generation
    Z_mb = random_generator(No, z_dim, dataT, Max_Seq_Len)
    # We pass the full padded dataset as input
    full_X_padded, _ = batch_generator(dataX_norm, dataT, range(No))
    X_hat_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: full_X_padded, T: dataT})
    
    # Extraction
    dataX_hat = []
    for i in range(No):
        Temp = X_hat_curr[i, :dataT[i], :]
        Temp = Temp * (max_val - min_val + 1e-7) + min_val
        dataX_hat.append(Temp)
    
    return dataX_hat
