import numpy as np
import matplotlib.pyplot as plt





def _heatmap(x_vals, y_vals, data, cmap, contour_values, contour_colors,
             rect_coords, fig_size, save_path):
    """Core heatmap drawing. data shape: (len(y_vals), len(x_vals))."""
    fig, ax = plt.subplots(figsize=fig_size)
    ax.pcolormesh(x_vals, y_vals, data, cmap=cmap, vmin=0, vmax=1)
    
    if contour_values:
        X, Y = np.meshgrid(x_vals, y_vals)
        ax.contour(X, Y, data, levels=contour_values, colors=contour_colors,
                   linewidths=1.5, alpha=0.8)
    
    if rect_coords is not None:
        x0, y0, w, h = rect_coords
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False,
                                    edgecolor='black', linewidth=2))
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('')
    ax.grid(False)
    fig.patch.set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def _infection_size(final_state, compartment_names):
    S_idx = compartment_names.index('S')
    data = 1.0 - final_state[..., S_idx, :].sum(axis=-1)
    if 'V' in compartment_names:
        V_idx = compartment_names.index('V')
        data = data - final_state[..., V_idx, :].sum(axis=-1)
    return np.asarray(data)


def plot_pol_homophily(final_state, compartment_names, pol_vals, h_vals,
                      cmap='viridis', contour_values=None, contour_colors='black',
                      rect_coords=None, fig_size=(5, 5),
                      xticks=None, yticks=None, xlim=None, ylim=None,
                      save_path=None):
    data = _infection_size(final_state, compartment_names).T  # (n_h, n_pol)
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.pcolormesh(pol_vals, h_vals, data, cmap=cmap, vmin=0, vmax=1)
    
    if contour_values:
        X, Y = np.meshgrid(pol_vals, h_vals)
        ax.contour(X, Y, data, levels=contour_values, colors=contour_colors,
                   linewidths=1.5, alpha=0.8)
    
    if rect_coords is not None:
        x0, y0, w, h = rect_coords
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False,
                                    edgecolor='black', linewidth=2))
    
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('')
    ax.grid(False)
    fig.patch.set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_pol_mean(final_state, compartment_names, pol_vals, mean_vals,
                  cmap='viridis', contour_values=None, contour_colors='white',
                  rect_coords=None, fig_size=(5, 5), save_path=None):
    """Publication heatmap of infection size over (polarization, mean).
    
    final_state: (len(pol_vals), len(mean_vals), n_comp, n_groups) from sweep_pol_mean
    """
    data = _infection_size(final_state, compartment_names)
    return _heatmap(pol_vals, mean_vals, data.T, cmap, contour_values, contour_colors,
                    rect_coords, fig_size, save_path)


def plot_extra_infections(beta_vals, gamma,
                          I_corners_M, I_base_M,
                          I_corners_T, I_base_T,
                          I_corners_V, I_base_V,
                          colors=('#7570b3', '#d95f02', '#1b9e77'),
                          labels=('SIR-M', 'SIR-T', 'SIR-V'),
                          fig_size=(3, 2), save_path=None):
    """Ribbon plot of extra infections (pp) vs R0 = beta / gamma."""
    R0s = np.asarray(beta_vals) / gamma
    LW = 0.5
    
    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    
    for (I_corners, I_base), color, label in zip(
        [(I_corners_M, I_base_M), (I_corners_T, I_base_T), (I_corners_V, I_base_V)],
        colors, labels,
    ):
        I_corners = np.asarray(I_corners)
        I_base = np.asarray(I_base)[:, None]
        delta = (I_corners - I_base) * 100
        lo, hi = delta.min(axis=1), delta.max(axis=1)
        axs.fill_between(R0s, lo, hi, color=color, alpha=0.8)
        print("maximum is: ", hi.max())
        axs.plot(R0s, lo, color='black', linewidth=LW, label=f'{label} min')
        axs.plot(R0s, hi, color='black', linewidth=LW, label=f'{label} max')
    
    axs.axhline(0, color='black', linestyle='--', linewidth=1)
    axs.axvline(2.5, color='black', linestyle=':', linewidth=1)
    
    axs.set_ylim(-50, 50)
    axs.set_xlim(1, 5)
    axs.set_xticks([1, 3, 5])
    axs.set_yticks([-50, 0, 50])
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    
    fig.patch.set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_I_band(I_corners, I_baseline, dT, color='#d95f02',
                fig_size=(3, 2), xlim=None, ylim=None,
                print_delta=True, save_path=None):
    """Plot I(t) baseline + min/max band over corner trajectories.
    
    If print_delta, also print final-time delta (band - baseline) in pp.
    """
    I_corners = np.asarray(I_corners)
    I_baseline = np.asarray(I_baseline)
    n_t = I_baseline.shape[0]
    t = np.arange(n_t) * dT
    
    lo = I_corners.min(axis=0)
    hi = I_corners.max(axis=0)
    
    if print_delta:
        d_lo = (lo[-1] - I_baseline[-1]) * 100
        d_hi = (hi[-1] - I_baseline[-1]) * 100
        avg = (d_lo + d_hi)*0.5
        print(f"average delta = {avg}")
    
    LW = 0.5
    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    axs.fill_between(t, lo, hi, color=color, alpha=1)
    axs.plot(t, lo, color='black', linewidth=LW)
    axs.plot(t, hi, color='black', linewidth=LW)
    axs.plot(t, I_baseline, color='black', linestyle='--', linewidth=1)
    
    if xlim is not None: axs.set_xlim(xlim)
    if ylim is not None: axs.set_ylim(ylim)
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_xticks([0,500,1000])
    axs.set_yticks([0, 0.5, 1])
    axs.set_xlim([-2,1002])
    axs.set_ylim([-0.01,1.01])
    
    fig.patch.set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig