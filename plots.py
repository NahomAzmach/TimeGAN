import numpy as np
import matplotlib.pyplot as plt

def smooth(values, window=25):
    values = np.asarray(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')

def loss_curves_figure(loss_history):
    """3-panel figure of the per-phase TimeGAN losses returned by timegan_torch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(smooth(loss_history['embedding_recon']))
    axes[0].set_title("1. Embedding: reconstruction MSE")
    axes[1].plot(smooth(loss_history['supervised']))
    axes[1].set_title("2. Supervisor: supervised MSE")
    for ax in axes[:2]:
        ax.set_yscale('log')
        ax.set_xlabel("Iteration")

    series = [('joint_D', "Discriminator"), ('joint_G_adversarial', "Generator (adversarial)"),
              ('joint_G_moments', "Generator (moments)")]
    if any(loss_history.get('joint_condition', [])):
        series.append(('joint_condition', "Generator (condition match)"))
    for key, label in series:
        axes[2].plot(smooth(loss_history[key]), label=label)
    # D loss is a sum of 3 BCE terms; 3·ln2 is what it would be if D couldn't tell real from fake at all
    axes[2].axhline(3 * np.log(2), color='gray', linestyle=':', label="D at chance (3·ln2)")
    axes[2].set_title("3. Joint training")
    axes[2].set_xlabel("Iteration")
    axes[2].legend(fontsize=8)
    return fig

LOSS_CAPTION = ("Curves are smoothed over 25 iterations. Phases 1–2 should drop and flatten before joint training starts. "
                "In phase 3, a discriminator loss far below the dotted line means it's winning and the generator isn't keeping up.")
