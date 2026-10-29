"""
PyTorch port of timegan_core.tgan (same interface, same losses and training schedule).

Differences from the TF1 version:
  - Plain nn.Module networks instead of tf.variable_scope graphs.
  - Returns per-phase loss history so training can be inspected, not just the final samples.
  - All sequences are assumed to have the same length (true for the 24-hour daily data here),
    so there is no sequence_length masking.

ConditionalTimeGAN extends it with a per-sequence condition vector c (e.g. peak hour, surge
strength, weekend, temperature). c is repeated over time and appended to the generator's noise
input and to the discriminator's input, so one model covers every scenario and the discriminator
penalizes samples that don't match the condition they were asked for.

On its own that wasn't enough: the discriminator could separate real from fake without checking
the condition, so the generator ignored surge strength entirely. Features that can be measured
from a curve (mean, peak hour, peak ratio) therefore also get a direct auxiliary loss: measure
them on each generated day (with a differentiable soft-argmax for peak hour) and penalize the
gap from what was requested.
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

def _with_condition(x, c):
    """Appends condition vector c [N, cond_dim] to every time step of x [N, T, D]."""
    if c is None:
        return x
    return torch.cat([x, c[:, None, :].expand(-1, x.shape[1], -1)], dim=2)

class TimeGAN(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_layers, z_dim, cond_dim=0):
        super().__init__()
        self.embedder = _RNNBlock(data_dim, hidden_dim, hidden_dim, num_layers)
        self.recovery = _RNNBlock(hidden_dim, hidden_dim, data_dim, num_layers)
        self.generator = _RNNBlock(z_dim + cond_dim, hidden_dim, hidden_dim, num_layers)
        self.supervisor = _RNNBlock(hidden_dim, hidden_dim, hidden_dim, num_layers - 1)
        self.discriminator = _RNNBlock(hidden_dim + cond_dim, hidden_dim, 1, num_layers, activation=None)

    def generate_latent(self, Z, c=None):
        return self.generator(_with_condition(Z, c))

    def discriminate(self, H, c=None):
        return self.discriminator(_with_condition(H, c))

    def generate(self, Z, c=None):
        return self.recovery(self.supervisor(self.generate_latent(Z, c)))

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
    device = _setup(parameters)

    No = len(dataX)
    data = np.array(dataX, dtype=np.float32)  # [No, Seq_Len, Data_Dim]
    seq_len, data_dim = data.shape[1], data.shape[2]

    # MinMax normalization over all sequences, per feature
    min_val = data.reshape(-1, data_dim).min(axis=0)
    max_val = data.reshape(-1, data_dim).max(axis=0)
    data_norm = torch.tensor((data - min_val) / (max_val - min_val + 1e-7), device=device)

    z_dim = parameters.get('z_dim', data_dim)
    model = TimeGAN(data_dim, parameters.get('hidden_dim', 24), parameters.get('num_layers', 3), z_dim).to(device)
    history = _fit(model, data_norm, None, parameters, z_dim, progress_callback)

    # Final Generation
    with torch.no_grad():
        X_hat = model.generate(torch.rand(No, seq_len, z_dim, device=device)).cpu().numpy()

    # Extraction: undo the normalization
    dataX_hat = [x * (max_val - min_val + 1e-7) + min_val for x in X_hat]

    return (dataX_hat, history) if return_history else dataX_hat

def _setup(parameters):
    if parameters.get('seed') is not None:
        torch.manual_seed(parameters['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # These GRUs are tiny, so multithreading costs more in coordination than it saves (~1.5x slower at 16 threads)
        torch.set_num_threads(1)
    return device

def _fit(model, data_norm, cond_norm, parameters, z_dim, progress_callback=None, condition_loss_fn=None):
    """
    The 3-phase TimeGAN training schedule, shared by the unconditional and conditional models.
    data_norm: [No, Seq_Len, Data_Dim] tensor in [0, 1]; cond_norm: [No, cond_dim] tensor or None.
    condition_loss_fn: optional (X_hat, C) -> scalar penalty for generated data not matching C.
    Returns the per-phase loss history.
    """
    No, seq_len = data_norm.shape[0], data_norm.shape[1]
    device = data_norm.device
    iterations = parameters.get('iterations', 5000)
    batch_size = min(parameters.get('batch_size', 128), No)
    gamma = 1
    shift = 1 if parameters.get('shifted_supervised_loss', True) else 0

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
        # Real sequences and their conditions; fakes in the same step reuse these conditions,
        # so the moments loss compares like with like
        idx = torch.randperm(No, device=device)[:batch_size]
        return data_norm[idx], (cond_norm[idx] if cond_norm is not None else None)

    def noise(n):
        return torch.rand(n, seq_len, z_dim, device=device)

    history = {name: [] for name in
               ["embedding_recon", "supervised", "joint_recon", "joint_supervised",
                "joint_G_adversarial", "joint_G_moments", "joint_D", "joint_condition"]}

    def supervised_loss(H):
        # Stepwise Supervised loss: supervisor predicts the next latent step from the current one
        H_hat_supervise = model.supervisor(H)
        if shift:
            return mse(H_hat_supervise[:, :-1, :], H[:, 1:, :])
        return mse(H_hat_supervise[:, 1:, :], H[:, 1:, :])

    # 1. Embedding Learning: train embedder + recovery as an autoencoder
    for itt in range(iterations):
        X, _ = batch()
        E_loss_T0 = mse(model.recovery(model.embedder(X)), X)
        E0_solver.zero_grad()
        (10 * torch.sqrt(E_loss_T0)).backward()
        E0_solver.step()
        history["embedding_recon"].append(E_loss_T0.item())
        if progress_callback and itt % 50 == 0:
            progress_callback("Embedding Network Training", itt, iterations)

    # 2. Training with Supervised Loss Only: teach the supervisor real latent dynamics
    for itt in range(iterations):
        X, _ = batch()
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
            X, C = batch()
            Z = noise(len(X))

            _trainable(model.embedder, model.recovery, model.discriminator, flag=False)
            with torch.no_grad():
                H = model.embedder(X)
            E_hat = model.generate_latent(Z, C)
            H_hat = model.supervisor(E_hat)
            X_hat = model.recovery(H_hat)
            Y_fake = model.discriminate(H_hat, C)
            Y_fake_e = model.discriminate(E_hat, C)

            G_loss_U = bce(Y_fake, torch.ones_like(Y_fake))
            G_loss_U_e = bce(Y_fake_e, torch.ones_like(Y_fake_e))
            G_loss_S = supervised_loss(H)
            G_loss_V = _moments_loss(X_hat, X)
            G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
            G_loss_C = condition_loss_fn(X_hat, C) if condition_loss_fn is not None else torch.zeros((), device=device)
            G_loss = G_loss + G_loss_C

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
        X, C = batch()
        Z = noise(len(X))
        with torch.no_grad():
            H = model.embedder(X)
            E_hat = model.generate_latent(Z, C)
            H_hat = model.supervisor(E_hat)
        _trainable(model.discriminator, flag=True)
        Y_real = model.discriminate(H, C)
        Y_fake = model.discriminate(H_hat, C)
        Y_fake_e = model.discriminate(E_hat, C)
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
        history["joint_condition"].append(G_loss_C.item())
        if progress_callback and itt % 50 == 0:
            progress_callback("Joint Training", itt, iterations)

    return history

class ConditionalTimeGAN:
    """
    One TimeGAN for every scenario: fit on all days with their condition vectors,
    then generate days for any condition vector.
    """

    def __init__(self, parameters, measurable_features=3):
        """
        parameters: TimeGAN parameters, plus 'condition_loss_weight' (default 50; 0 disables the
                    auxiliary loss and leaves conditioning to the discriminator alone).
        measurable_features: the first n condition columns are [mean, peak_hour, peak_ratio],
                             measurable from a generated curve (see data_processor.SHAPE_FEATURES).
        """
        self.parameters = parameters
        self.measurable_features = measurable_features
        self.model = None
        self.history = None

    def fit(self, dataX, conditions, progress_callback=None):
        """
        dataX: [No, Seq_Len, Data_Dim] array; conditions: [No, cond_dim] raw feature values.
        """
        self.device = _setup(self.parameters)
        data = np.asarray(dataX, dtype=np.float32)
        conditions = np.asarray(conditions, dtype=np.float32)
        self.seq_len, data_dim = data.shape[1], data.shape[2]

        # Data and conditions are both MinMax-scaled to [0, 1] per feature
        self.min_val = data.reshape(-1, data_dim).min(axis=0)
        self.max_val = data.reshape(-1, data_dim).max(axis=0)
        self.cond_min = conditions.min(axis=0)
        self.cond_max = conditions.max(axis=0)
        data_norm = torch.tensor((data - self.min_val) / (self.max_val - self.min_val + 1e-7), device=self.device)
        cond_norm = self._scale_conditions(conditions)

        self.z_dim = self.parameters.get('z_dim', data_dim)
        self.model = TimeGAN(data_dim, self.parameters.get('hidden_dim', 24), self.parameters.get('num_layers', 3),
                             self.z_dim, cond_dim=conditions.shape[1]).to(self.device)
        weight = self.parameters.get('condition_loss_weight', 50)
        condition_loss_fn = self._condition_loss if weight > 0 and self.measurable_features > 0 else None
        self._condition_loss_weight = weight
        self.history = _fit(self.model, data_norm, cond_norm, self.parameters, self.z_dim,
                            progress_callback, condition_loss_fn)
        return self

    def _measure(self, X_hat):
        """
        Differentiable [mean, peak_hour, peak_ratio] of generated curves, in raw units.
        X_hat: [N, T, 1] normalized output of the recovery network.
        """
        scale = torch.tensor(self.max_val - self.min_val + 1e-7, device=self.device)
        offset = torch.tensor(self.min_val, device=self.device)
        x = (X_hat * scale + offset)[:, :, 0]
        mean = x.mean(dim=1)
        # Soft-argmax: a temperature-sharpened softmax over each day's standardized curve
        z = (x - mean[:, None]) / (x.std(dim=1, keepdim=True) + 1e-6)
        hours = torch.arange(x.shape[1], device=self.device, dtype=x.dtype)
        peak_hour = (torch.softmax(z * 5.0, dim=1) * hours).sum(dim=1)
        peak_ratio = x.max(dim=1).values / mean.clamp(min=1e-3)
        return torch.stack([mean, peak_hour, peak_ratio], dim=1)[:, :self.measurable_features]

    def _condition_loss(self, X_hat, C):
        # Compare in the same [0, 1] scale as the conditions so every feature weighs about equally
        n = self.measurable_features
        cmin = torch.tensor(self.cond_min[:n], device=self.device)
        crange = torch.tensor(self.cond_max[:n] - self.cond_min[:n] + 1e-7, device=self.device)
        measured = (self._measure(X_hat) - cmin) / crange
        return self._condition_loss_weight * torch.mean(torch.abs(measured - C[:, :n]))

    def _scale_conditions(self, conditions):
        conditions = np.asarray(conditions, dtype=np.float32)
        return torch.tensor((conditions - self.cond_min) / (self.cond_max - self.cond_min + 1e-7), device=self.device)

    def generate(self, conditions):
        """conditions: [N, cond_dim] raw feature values. Returns [N, Seq_Len, Data_Dim] sequences."""
        c = self._scale_conditions(conditions)
        with torch.no_grad():
            Z = torch.rand(len(c), self.seq_len, self.z_dim, device=self.device)
            X_hat = self.model.generate(Z, c).cpu().numpy()
        return X_hat * (self.max_val - self.min_val + 1e-7) + self.min_val
