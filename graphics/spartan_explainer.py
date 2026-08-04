import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import jax
import jax.numpy as jnp
import cosmographi as cg
import numpy as np


def generate_samples(
    Nsim, Mmean, Mstd, m_threshold, m_std, z_std, z_max, Omega_m, w0, wa=0.0, key=None
):
    cosmology = cg.Cosmology()
    cosmology.Omega_m = Omega_m
    cosmology.w0 = w0
    cosmology.wa = wa
    rateIa = cg.rates.RateConst(cosmology, r=1, z_min=0.0, z_max=z_max, name="rateIa")

    # Sample true redshifts
    # --------------------------------------------------------------------
    key, subkey = jax.random.split(key)
    z_true = rateIa.sample_z(subkey, Nsim)

    # Compute true distance modulus
    # --------------------------------------------------------------------
    DL = jax.jit(jax.vmap(cosmology.luminosity_distance))(z_true)
    mu_true = 5 * jnp.log10(DL) - 5

    # Sample true absolute magnitudes
    # --------------------------------------------------------------------
    key, subkey = jax.random.split(key)
    M_true = jax.random.normal(subkey, shape=(Nsim,)) * Mstd + Mmean

    # Compute true apparent magnitudes
    # --------------------------------------------------------------------
    m_true = mu_true + M_true

    # Generate noised values with noise
    # --------------------------------------------------------------------
    z_noise = z_true + jax.random.normal(key, shape=(Nsim,)) * z_std
    m_noise = m_true + jax.random.normal(key, shape=(Nsim,)) * m_std

    # Apply detection threshold
    # --------------------------------------------------------------------
    select = m_noise < m_threshold
    z_sel = z_noise[select]
    m_sel = m_noise[select]

    return {
        "z_true": z_true,
        "mu_true": mu_true,
        "m_true": m_true,
        "M_true": M_true,
        "Mmean": Mmean,
        "z_noise": z_noise,
        "m_noise": m_noise,
        "z_sel": z_sel,
        "m_sel": m_sel,
        "select": select,
        "Mstd": Mstd,
        "Omega_m": Omega_m,
        "w0": w0,
        "wa": wa,
    }


def importance_weights(samples, Omega_m, w0, wa=0.0):
    # Compute the distance modulus for the new cosmology
    cosmo_new = cg.Cosmology()
    cosmo_new.Omega_m = Omega_m
    cosmo_new.w0 = w0
    cosmo_new.wa = wa

    DL_new = jax.jit(jax.vmap(cosmo_new.luminosity_distance))(samples["z_true"])
    mu_new = 5 * jnp.log10(DL_new) - 5

    # Compute the importance weights
    weights_denom = -0.5 * (samples["M_true"] - samples["Mmean"]) ** 2 / samples["Mstd"] ** 2
    weights_numer = (
        -0.5 * (samples["mu_true"] - mu_new + samples["M_true"] - samples["Mmean"]) ** 2 / 0.1**2
    )

    return jnp.exp(weights_numer - weights_denom), mu_new


def main():
    # Build a 2x3 parent grid where upper-middle and upper-right are each split into 2x1.
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)

    parent = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.05,
        hspace=0.0,
    )

    # Main-level panels.
    ax_ul = fig.add_subplot(parent[0, 0])
    ax_ll = fig.add_subplot(parent[1, 0])
    ax_lm = fig.add_subplot(parent[1, 1])
    ax_lr = fig.add_subplot(parent[1, 2])

    # Upper-middle panel split into a nested 2x1 grid.
    nested_um = parent[0, 1].subgridspec(
        nrows=2,
        ncols=1,
        wspace=0.04,
        hspace=0.04,
    )
    ax_ur_00 = fig.add_subplot(nested_um[0, 0])
    ax_ur_10 = fig.add_subplot(nested_um[1, 0])

    # Upper-right panel split into a nested 2x1 grid.
    nested_ur = parent[0, 2].subgridspec(
        nrows=2,
        ncols=1,
        wspace=0.04,
        hspace=0.04,
    )
    ax_ur_01 = fig.add_subplot(nested_ur[0, 0])
    ax_ur_11 = fig.add_subplot(nested_ur[1, 0])

    # Upper left figure
    # ----------------------------------------------------------------
    print("sims")
    sims = generate_samples(
        Nsim=2000,
        Mmean=-19.3,
        Mstd=0.3,
        m_threshold=-4,
        m_std=0.1,
        z_std=0.05,
        z_max=2.0,
        Omega_m=0.3,
        w0=-1.0,
        wa=0.0,
        key=jax.random.PRNGKey(42),
    )
    xlims = (jnp.min(sims["z_true"]), jnp.max(sims["z_true"]))
    ylims = (jnp.min(sims["m_true"]), jnp.max(sims["m_true"]))
    ax_ul.scatter(sims["z_true"], sims["m_true"], s=5, c="k", linewidth=0)
    ax_ul.plot(
        jnp.sort(sims["z_true"]),
        sims["Mmean"] + sims["mu_true"][jnp.argsort(sims["z_true"])],
        c="tab:orange",
        linestyle="--",
        lw=2,
    )
    ax_ul.set_xlim(xlims)
    ax_ul.set_ylim(ylims)
    ax_ul.set_xlabel("Redshift (z)")
    ax_ul.set_ylabel("Apparent Magnitude (m)")
    ax_ul.set_title("Fiducial Sample: $D_i \\sim P(D_i|\\Omega^*)$")

    # Upper middle figure
    # ----------------------------------------------------------------
    # (Now occupied by two stacked subpanels.)

    # add lines connecting selected samples
    sim_points = []
    connect_ax = []
    connect_ax.append(ax_ur_00)
    sim_points.append((sims["z_true"][~sims["select"]][0], sims["m_true"][~sims["select"]][0]))
    connect_ax.append(ax_ur_01)
    sim_points.append((sims["z_true"][~sims["select"]][1], sims["m_true"][~sims["select"]][1]))
    connect_ax.append(ax_ur_10)
    sim_points.append((sims["z_true"][sims["select"]][0], sims["m_true"][sims["select"]][0]))
    connect_ax.append(ax_ur_11)
    sim_points.append((sims["z_true"][sims["select"]][1], sims["m_true"][sims["select"]][1]))
    for sim, ax_lc in zip(sim_points, connect_ax):
        connector = ConnectionPatch(
            xyA=sim,
            coordsA=ax_ul.transData,
            xyB=(0.0, 0.95),
            coordsB=ax_lc.transAxes,
            color="tab:gray",
            linestyle="--",
            lw=1.5,
            # alpha=0.5,
            clip_on=False,
            # zorder=10,
        )
        fig.add_artist(connector)

    # Upper right figures
    # ----------------------------------------------------------------
    print("Selection examples")
    x = np.linspace(-1, 1, 7)
    np.random.seed(43)
    for subax, scale in zip([ax_ur_00, ax_ur_01, ax_ur_10, ax_ur_11], [0.8, 0.1, 1.4, 2.5]):
        subax.errorbar(
            x,
            scale * np.exp(-0.5 * x**2 / 0.5**2) + np.random.normal(scale=0.2, size=len(x)),
            yerr=0.2,
            fmt="o",
            c="tab:green" if scale > 1 else "tab:red",
        )
        subax.axhline(1.0, linestyle="--", c="grey")
        subax.set_xlim([-1.05, 1.05])
        subax.set_ylim([0, 3])
        subax.set_xticks([])
        subax.set_yticks([])
    ax_ur_00.set_title("Not selected: $\\mathbb{S}_i = 0$")
    ax_ur_01.set_title("Not selected: $\\mathbb{S}_i = 0$")
    ax_ur_10.set_title("Selected: $\\mathbb{S}_i = 1$")
    ax_ur_11.set_title("Selected: $\\mathbb{S}_i = 1$")

    # Lower left figure
    # ----------------------------------------------------------------
    print("Selected sims")
    ax_ll.scatter(
        sims["z_true"][sims["select"]],
        sims["m_true"][sims["select"]],
        s=5,
        # alpha=0.5,
        c="tab:green",
        linewidth=0,
    )
    ax_ll.plot(
        jnp.sort(sims["z_true"]),
        sims["Mmean"] + sims["mu_true"][jnp.argsort(sims["z_true"])],
        c="tab:orange",
        linestyle="--",
        lw=2,
    )
    ax_ll.set_xlim(xlims)
    ax_ll.set_ylim(ylims)
    ax_ll.set_xlabel("Redshift (z)")
    ax_ll.set_ylabel("Apparent Magnitude (m)")
    ax_ll.set_title("Selected Sample: $P(\\mathbb{S}_i|D_i)$")

    # Lower middle figure
    # ----------------------------------------------------------------
    Om_ref = 0.36
    w0_ref = -0.8
    print("importance weights")
    weights, mu_new = importance_weights(sims, Omega_m=Om_ref, w0=w0_ref, wa=0.0)
    weights = weights[sims["select"]]
    weights = weights / jnp.max(weights)  # Normalize weights for plotting
    ax_lm.scatter(
        sims["z_true"][sims["select"]],
        sims["m_true"][sims["select"]],
        s=5,
        alpha=(weights + 0.2) / 1.2,
        c="tab:green",
        linewidth=0,
    )
    ax_lm.plot(
        jnp.sort(sims["z_true"]),
        sims["Mmean"] + sims["mu_true"][jnp.argsort(sims["z_true"])],
        c="tab:orange",
        linestyle="--",
        lw=2,
    )
    ax_lm.plot(
        jnp.sort(sims["z_true"]),
        sims["Mmean"] + mu_new[jnp.argsort(sims["z_true"])],
        c="tab:blue",
        lw=2,
    )
    ax_lm.set_xlim(xlims)
    ax_lm.set_ylim(ylims)
    ax_lm.set_xlabel("Redshift (z)")
    ax_lm.set_ylabel("Apparent Magnitude (m)")
    ax_lm.set_title("Importance Weights: $\\frac{P(D_i|\\Omega)}{P(D_i|\\Omega^*)}$")
    # Lower right figure
    # ----------------------------------------------------------------
    print("Heatmap")
    Om, W0 = jnp.meshgrid(jnp.linspace(0.2, 0.4, 100), jnp.linspace(-1.3, -0.7, 100))
    Z = jax.vmap(lambda om, w0: jnp.sum(importance_weights(sims, Omega_m=om, w0=w0)[0]))(
        Om.ravel(), W0.ravel()
    )
    Z = Z.reshape(Om.shape)
    ax_lr.contourf(Om, W0, Z, levels=25, cmap="Greys")
    ax_lr.scatter(0.3, -1.0, c="tab:orange", marker="*", s=125)
    ax_lr.scatter(Om_ref, w0_ref, c="tab:blue", marker="o", s=100, zorder=21)

    # Connect the lower-middle panel to the corresponding blue point in parameter space.
    blue_point = (Om_ref, w0_ref)
    for corner in [(1.0, 0.0), (1.0, 1.0)]:
        connector = ConnectionPatch(
            xyA=corner,
            coordsA=ax_lm.transAxes,
            xyB=blue_point,
            coordsB=ax_lr.transData,
            color="tab:gray",
            linestyle="--",
            lw=1.5,
            alpha=0.5,
            clip_on=False,
            zorder=10,
        )
        fig.add_artist(connector)

    ax_lr.set_xlabel("Omega_m")
    ax_lr.set_ylabel("w0")
    ax_lr.set_title("Marginal Detection Probability: $P(\\mathbb{S}|\\Omega)$")

    # Constrained layout plus per-axis titles can make stacked subpanels slightly taller.
    # Squish all top-row subpanels so their top/bottom align with the top-left panel.
    fig.canvas.draw()
    ur_axes = [ax_ur_00, ax_ur_01, ax_ur_10, ax_ur_11]
    top_ref = ax_ul.get_position()
    ur_y0 = min(ax.get_position().y0 for ax in ur_axes)
    ur_y1 = max(ax.get_position().y1 for ax in ur_axes)

    target_y0 = top_ref.y0
    target_y1 = top_ref.y1
    current_h = ur_y1 - ur_y0
    target_h = target_y1 - target_y0

    # Affine remap each nested axis y-position into the target vertical span.
    if current_h > 0:
        yscale = target_h / current_h
        for ax in ur_axes:
            pos = ax.get_position()
            new_y0 = target_y0 + (pos.y0 - ur_y0) * yscale
            new_h = pos.height * yscale
            ax.set_position([pos.x0, new_y0, pos.width, new_h])

    plt.savefig("spartan_explainer.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
