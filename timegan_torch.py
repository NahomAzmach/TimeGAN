"""
PyTorch port of timegan_core.tgan (same interface, same losses and training schedule).

Differences from the TF1 version:
  - Plain nn.Module networks instead of tf.variable_scope graphs.
  - Returns per-phase loss history so training can be inspected, not just the final samples.
  - All sequences are assumed to have the same length (true for the 24-hour daily data here),
    so there is no sequence_length masking.
"""
import numpy as np
import torch
from torch import nn

class _RNNBlock(nn.Module):
    """Stacked GRU followed by a linear layer, the building block of all five TimeGAN networks."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=torch.sigmoid):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.activation = activation
        self._tf_style_init()

    def _tf_style_init(self):
        # Match TF1's GRUCell/dense defaults (Glorot-uniform weights, gate biases of 1, other
        # biases 0) instead of PyTorch's small uniform init; without this the port trains worse.
        hidden = self.rnn.hidden_size
        for name, param in self.rnn.named_parameters():
            if name.startswith("weight"):
                for gate in param.data.chunk(3, dim=0):  # reset, update, candidate
                    nn.init.xavier_uniform_(gate)
            elif name.startswith("bias_ih"):
                nn.init.zeros_(param)
                nn.init.ones_(param.data[:2 * hidden])  # reset and update gates start open
            else:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out)
        return self.activation(out) if self.activation is not None else out

class TimeGAN(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_layers, z_dim):
        super().__init__()
        self.embedder = _RNNBlock(data_dim, hidden_dim, hidden_dim, num_layers)
        self.recovery = _RNNBlock(hidden_dim, hidden_dim, data_dim, num_layers)
        self.generator = _RNNBlock(z_dim, hidden_dim, hidden_dim, num_layers)
        self.supervisor = _RNNBlock(hidden_dim, hidden_dim, hidden_dim, num_layers - 1)
        self.discriminator = _RNNBlock(hidden_dim, hidden_dim, 1, num_layers, activation=None)

    def generate(self, Z):
        return self.recovery(self.supervisor(self.generator(Z)))

def _moments_loss(X_hat, X):
    # Two Moments Loss (Mean and Variance constraint), statistics taken over the batch axis
    std_hat = torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
    std = torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)
    G_loss_V1 = torch.mean(torch.abs(std_hat - std))
    G_loss_V2 = torch.mean(torch.abs(X_hat.mean(dim=0) - X.mean(dim=0)))
    return G_loss_V1 + G_loss_V2

def device_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def _trainable(*modules, flag):
    # Freezing networks that a step doesn't update skips their weight gradients,
    # which TF got for free from var_list. Gradients still flow *through* them.
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(flag)

def tgan(dataX, parameters, progress_callback=None, return_history=False):
    """
    TimeGAN Generator in PyTorch.
    dataX: List of sequences, e.g. [np.array( [Seq_Len, Data_Dim] ), ...]
    parameters: dict of TimeGAN parameters (hidden_dim, num_layers, iterations, batch_size, z_dim, seed)
                'shifted_supervised_loss' (default True) uses the paper's next-step supervised loss;
                False reproduces timegan_core.py, which compares H and S(H) at the same step.
    progress_callback: function(phase, current_it, total_it) to update the Streamlit UI
    return_history: if True, returns (dataX_hat, history) where history maps loss name -> list
    """
    if parameters.get('seed') is not None:
        torch.manual_seed(parameters['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # These GRUs are tiny, so multithreading costs more in coordination than it saves (~1.5x slower at 16 threads)
        torch.set_num_threads(1)

    No = len(dataX)
    data = np.array(dataX, dtype=np.float32)  # [No, Seq_Len, Data_Dim]
    seq_len, data_dim = data.shape[1], data.shape[2]

    # MinMax normalization over all sequences, per feature
    min_val = data.reshape(-1, data_dim).min(axis=0)
    max_val = data.reshape(-1, data_dim).max(axis=0)
    data_norm = torch.tensor((data - min_val) / (max_val - min_val + 1e-7), device=device)

    hidden_dim = parameters.get('hidden_dim', 24)
    num_layers = parameters.get('num_layers', 3)
    iterations = parameters.get('iterations', 5000)
    batch_size = min(parameters.get('batch_size', 128), No)
    z_dim = parameters.get('z_dim', data_dim)
    gamma = 1
    shift = 1 if parameters.get('shifted_supervised_loss', True) else 0

    model = TimeGAN(data_dim, hidden_dim, num_layers, z_dim).to(device)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    er_params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    gs_params = list(model.generator.parameters()) + list(model.supervisor.parameters())
    E0_solver = torch.optim.Adam(er_params)
    E_solver = torch.optim.Adam(er_params)
    D_solver = torch.optim.Adam(model.discriminator.parameters())
    G_solver = torch.optim.Adam(gs_params)
    GS_solver = torch.optim.Adam(gs_params)

    def batch():
        idx = torch.randperm(No, device=device)[:batch_size]
        return data_norm[idx]

    def noise(n):
        return torch.rand(n, seq_len, z_dim, device=device)

    history = {name: [] for name in
               ["embedding_recon", "supervised", "joint_recon", "joint_supervised",
                "joint_G_adversarial", "joint_G_moments", "joint_D"]}

    def supervised_loss(H):
        # Stepwise Supervised loss: supervisor predicts the next latent step from the current one
        H_hat_supervise = model.supervisor(H)
        if shift:
            return mse(H_hat_supervise[:, :-1, :], H[:, 1:, :])
        return mse(H_hat_supervise[:, 1:, :], H[:, 1:, :])

    # 1. Embedding Learning: train embedder + recovery as an autoencoder
    for itt in range(iterations):
        X = batch()
        E_loss_T0 = mse(model.recovery(model.embedder(X)), X)
        E0_solver.zero_grad()
        (10 * torch.sqrt(E_loss_T0)).backward()
        E0_solver.step()
        history["embedding_recon"].append(E_loss_T0.item())
        if progress_callback and itt % 50 == 0:
            progress_callback("Embedding Network Training", itt, iterations)

    # 2. Training with Supervised Loss Only: teach the supervisor real latent dynamics
    for itt in range(iterations):
        X = batch()
        with torch.no_grad():
            H = model.embedder(X)
        G_loss_S = supervised_loss(H)
        GS_solver.zero_grad()
        G_loss_S.backward()
        GS_solver.step()
        history["supervised"].append(G_loss_S.item())
        if progress_callback and itt % 50 == 0:
            progress_callback("Supervised Loss Training", itt, iterations)

    # 3. Joint Training
    for itt in range(iterations):
        # Generator Training (twice)
        for kk in range(2):
            X = batch()
            Z = noise(len(X))

            _trainable(model.embedder, model.recovery, model.discriminator, flag=False)
            with torch.no_grad():
                H = model.embedder(X)
            E_hat = model.generator(Z)
            H_hat = model.supervisor(E_hat)
            X_hat = model.recovery(H_hat)
            Y_fake = model.discriminator(H_hat)
            Y_fake_e = model.discriminator(E_hat)

            G_loss_U = bce(Y_fake, torch.ones_like(Y_fake))
            G_loss_U_e = bce(Y_fake_e, torch.ones_like(Y_fake_e))
            G_loss_S = supervised_loss(H)
            G_loss_V = _moments_loss(X_hat, X)
            G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

            G_solver.zero_grad()
            G_loss.backward()
            G_solver.step()
            _trainable(model.embedder, model.recovery, flag=True)
            _trainable(model.generator, model.supervisor, flag=False)

            # Embedder network loss: reconstruction plus a little supervised loss
            H = model.embedder(X)
            E_loss_T0 = mse(model.recovery(H), X)
            E_loss = 10 * torch.sqrt(E_loss_T0) + 0.1 * supervised_loss(H)
            E_solver.zero_grad()
            E_loss.backward()
            E_solver.step()
            _trainable(model.generator, model.supervisor, flag=True)

        # Discriminator Training
        X = batch()
        Z = noise(len(X))
        with torch.no_grad():
            H = model.embedder(X)
            E_hat = model.generator(Z)
            H_hat = model.supervisor(E_hat)
        _trainable(model.discriminator, flag=True)
        Y_real = model.discriminator(H)
        Y_fake = model.discriminator(H_hat)
        Y_fake_e = model.discriminator(E_hat)
        D_loss = bce(Y_real, torch.ones_like(Y_real)) + \
                 bce(Y_fake, torch.zeros_like(Y_fake)) + \
                 gamma * bce(Y_fake_e, torch.zeros_like(Y_fake_e))

        # Only update the discriminator when it's not already winning, to keep it from overpowering G
        if D_loss.item() > 0.15:
            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step()

        history["joint_recon"].append(E_loss_T0.item())
        history["joint_supervised"].append(G_loss_S.item())
        history["joint_G_adversarial"].append(G_loss_U.item())
        history["joint_G_moments"].append(G_loss_V.item())
        history["joint_D"].append(D_loss.item())
        if progress_callback and itt % 50 == 0:
            progress_callback("Joint Training", itt, iterations)

    # Final Generation
    with torch.no_grad():
        X_hat = model.generate(noise(No)).cpu().numpy()

    # Extraction: undo the normalization
    dataX_hat = [x * (max_val - min_val + 1e-7) + min_val for x in X_hat]

    return (dataX_hat, history) if return_history else dataX_hat
