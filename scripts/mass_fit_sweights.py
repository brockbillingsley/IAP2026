import os
import json
import numpy as np
import matplotlib.pyplot as plt

import zfit

# Try core zfit first; otherwise use zfit_physics
try:
    GeneralizedCB = zfit.pdf.GeneralizedCB
except AttributeError:
    from zfit_physics.pdf import GeneralizedCB

from hepstats.splot import compute_sweights

def fit_shape_unbinned(pdf, mass_array, obs, label="shape"):
    data = zfit.Data.from_numpy(obs=obs, array=mass_array)
    loss = zfit.loss.UnbinnedNLL(model=pdf, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss)
    result.hesse()
    print(f"{label} fit: converged={result.converged} valid={getattr(result, 'valid', None)}")
    return result

# -----------------------------
# 1) Model building
# -----------------------------
def build_mass_model(B_mass_obs, n_total_guess):
    # Signal params (tune starting values/ranges to mass window)
    mu     = zfit.Parameter("mu",     5.279, 5.24,  5.32)

    sigmal = zfit.Parameter("sigmal", 0.020, 0.005, 0.060)
    alphal = zfit.Parameter("alphal", 1.5,   0.01,  10.0)
    nl     = zfit.Parameter("nl",     3.0,   0.1,   50.0)

    sigmar = zfit.Parameter("sigmar", 0.020, 0.005, 0.060)
    alphar = zfit.Parameter("alphar", 1.5,   0.01,  10.0)
    nr     = zfit.Parameter("nr",     3.0,   0.1,   50.0)

    pdfsig = GeneralizedCB(
        obs=B_mass_obs,
        mu=mu,
        sigmal=sigmal, alphal=alphal, nl=nl,
        sigmar=sigmar, alphar=alphar, nr=nr,
    )

    # Background model
    lambda_bkg = zfit.Parameter("lambda_bkg", -2.0, -50.0, 50.0)  # allow +/-
    pdfbkg = zfit.pdf.Exponential(obs=B_mass_obs, lambda_=lambda_bkg)


    # Yields (must be zfit Parameters)
    Nsig = zfit.Parameter("Nsig", 0.5 * n_total_guess, 0.0, 1.2 * n_total_guess)
    Nbkg = zfit.Parameter("Nbkg", 0.5 * n_total_guess, 0.0, 1.2 * n_total_guess)


    # Extend
    sig_ext = pdfsig.create_extended(Nsig)
    bkg_ext = pdfbkg.create_extended(Nbkg)

    # Sum
    model = zfit.pdf.SumPDF([sig_ext, bkg_ext])
    return model, sig_ext, bkg_ext, (Nsig, Nbkg)


# -----------------------------
# 2) Fit
# -----------------------------
def fit_mass(model, mass_array, B_mass_obs):
    data = zfit.Data.from_numpy(obs=B_mass_obs, array=mass_array)
    loss = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss)
    result.hesse()
    return result, data


def save_fitresult(result, out_json):
    payload = {}
    for p, info in result.params.items():
        payload[p.name] = {
            "value": float(info["value"]),
            "error": float(info["hesse"]["error"]) if "hesse" in info else None,
        }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)


# -----------------------------
# 3) Plot mass fit
# -----------------------------
def plot_mass_fit(mass_array, B_mass_obs, model, sig_ext, bkg_ext, out_png, nbins=60):
    lims = np.asarray(B_mass_obs.limits, dtype=float).reshape(-1)
    lo = float(np.min(lims))
    hi = float(np.max(lims))



    counts, edges = np.histogram(mass_array, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    x = zfit.Data.from_numpy(obs=B_mass_obs, array=centers)

    y_tot = np.array(zfit.run(model.ext_pdf(x))) * widths
    y_sig = np.array(zfit.run(sig_ext.ext_pdf(x))) * widths
    y_bkg = np.array(zfit.run(bkg_ext.ext_pdf(x))) * widths

    plt.figure()
    plt.errorbar(centers, counts, yerr=np.sqrt(np.maximum(counts, 1.0)), fmt=".", label="data")
    plt.plot(centers, y_tot, label="total")
    plt.plot(centers, y_sig, label="signal")
    plt.plot(centers, y_bkg, label="background")
    plt.xlabel("B_mass")
    plt.ylabel("Counts / bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# 4) Per-angle sWeight checks
# -----------------------------
def plot_sweight_checks_per_angle(
    angles_mix, angles_sig_true, angles_bkg_true,
    w_sig, w_bkg,
    angle_names,
    out_dir,
    nbins=40,
):
    os.makedirs(out_dir, exist_ok=True)

    for ang in angle_names:
        x_mix = angles_mix[ang]
        x_sig = angles_sig_true[ang]
        x_bkg = angles_bkg_true[ang]

        lo = float(np.nanmin(np.concatenate([x_mix, x_sig, x_bkg])))
        hi = float(np.nanmax(np.concatenate([x_mix, x_sig, x_bkg])))

        fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        axs = axs.flatten()

        axs[0].hist(x_mix, bins=nbins, range=(lo, hi), weights=w_sig)
        axs[0].set_title(f"{ang}: combined × signal sWeight")

        axs[1].hist(x_mix, bins=nbins, range=(lo, hi), weights=w_bkg)
        axs[1].set_title(f"{ang}: combined × background sWeight")

        axs[2].hist(x_sig, bins=nbins, range=(lo, hi))
        axs[2].set_title(f"{ang}: truth signal only")

        axs[3].hist(x_bkg, bins=nbins, range=(lo, hi))
        axs[3].set_title(f"{ang}: truth background only")

        axs[2].set_xlabel(ang)
        axs[3].set_xlabel(ang)
        for ax in axs:
            ax.set_ylabel("Entries")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sweight_check_{ang}.png"), dpi=200)
        plt.close(fig)


# -----------------------------
# MAIN
# -----------------------------
def main():

    import uproot

    def load_root_sample(root_path, tree_name, branches):
        with uproot.open(root_path) as f:
            tree = f[tree_name]
            arr = tree.arrays(branches, library="np")
        return arr

    # -------------------------
    # LOAD DATA HERE
    # -------------------------
    sig_path = "scripts/sig_toy.root"
    bkg_path = "genbkg/bkg_toy.root"

    sig_tree = "signal"
    bkg_tree = "background"

    branches = ["B_mass", "cosThetaK", "cosThetaL"]

    arr_sig = load_root_sample(sig_path, sig_tree, branches)
    arr_bkg = load_root_sample(bkg_path, bkg_tree, branches)

    mass_sig = arr_sig["B_mass"]
    mass_bkg = arr_bkg["B_mass"]

    angles_sig_true = {
        "cosThetaK": arr_sig["cosThetaK"],
        "cosThetaL": arr_sig["cosThetaL"],
    }
    angles_bkg_true = {
        "cosThetaK": arr_bkg["cosThetaK"],
        "cosThetaL": arr_bkg["cosThetaL"],
    }


    # ---- Step 3 sanity checks ----
    assert len(mass_sig) == len(angles_sig_true["cosThetaK"]) == len(angles_sig_true["cosThetaL"])
    assert len(mass_bkg) == len(angles_bkg_true["cosThetaK"]) == len(angles_bkg_true["cosThetaL"])

    def finite(x): return np.isfinite(x).all()
    assert finite(mass_sig) and finite(mass_bkg)
    assert finite(angles_sig_true["cosThetaK"]) and finite(angles_sig_true["cosThetaL"])
    assert finite(angles_bkg_true["cosThetaK"]) and finite(angles_bkg_true["cosThetaL"])

    print("Loaded:")
    print("  signal events:", len(mass_sig), "mass range:", mass_sig.min(), mass_sig.max())
    print("  bkg events:   ", len(mass_bkg), "mass range:", mass_bkg.min(), mass_bkg.max())

    # -------------------------
    # Define mass observable
    # -------------------------
    B_mass = zfit.Space("B_mass", limits=(5.170, 5.500))

    # -------------------------
    # Build SHAPE PDFs (not extended) for prefit
    # -------------------------
    # Signal params (same as in build_mass_model)
    mu     = zfit.Parameter("mu",     5.279, 5.24,  5.32)

    sigmal = zfit.Parameter("sigmal", 0.020, 0.005, 0.060)
    alphal = zfit.Parameter("alphal", 1.5,   0.05,  10.0)
    nl     = zfit.Parameter("nl",     3.0,   0.5,   50.0)

    sigmar = zfit.Parameter("sigmar", 0.020, 0.005, 0.060)
    alphar = zfit.Parameter("alphar", 1.5,   0.05,  10.0)
    nr     = zfit.Parameter("nr",     3.0,   0.5,   50.0)

    pdfsig = GeneralizedCB(
        obs=B_mass,
        mu=mu,
        sigmal=sigmal, alphal=alphal, nl=nl,
        sigmar=sigmar, alphar=alphar, nr=nr,
    )

    # Background exponential: allow both signs
    lambda_bkg = zfit.Parameter("lambda_bkg", -2.0, -50.0, 50.0)
    pdfbkg = zfit.pdf.Exponential(obs=B_mass, lambda_=lambda_bkg)

    # -------------------------
    # Prefit shapes on pure samples
    # -------------------------
    result_sig_shape = fit_shape_unbinned(pdfsig, mass_sig, B_mass, label="signal shape")
    result_bkg_shape = fit_shape_unbinned(pdfbkg, mass_bkg, B_mass, label="bkg shape")

    print("Prefit lambda_bkg:", float(lambda_bkg.value()))

    # -------------------------
    # Freeze shape parameters for stable sWeights
    # -------------------------
    for p in [mu, sigmal, alphal, nl, sigmar, alphar, nr, lambda_bkg]:
        p.floating = False

    # -------------------------
    # Build combined dataset
    # -------------------------
    mass_mix = np.concatenate([mass_sig, mass_bkg])
    angles_mix = {
        "cosThetaK": np.concatenate([angles_sig_true["cosThetaK"], angles_bkg_true["cosThetaK"]]),
        "cosThetaL": np.concatenate([angles_sig_true["cosThetaL"], angles_bkg_true["cosThetaL"]]),
    }
    assert len(angles_mix["cosThetaK"]) == len(mass_mix)
    assert len(angles_mix["cosThetaL"]) == len(mass_mix)


    # -------------------------
    # Extended yields-only fit
    # -------------------------
    Ntot = len(mass_mix)
    Nsig = zfit.Parameter("Nsig", len(mass_sig), 0.0, 1.2 * Ntot)
    Nbkg = zfit.Parameter("Nbkg", len(mass_bkg), 0.0, 1.2 * Ntot)

    sig_ext = pdfsig.create_extended(Nsig)
    bkg_ext = pdfbkg.create_extended(Nbkg)
    model = zfit.pdf.SumPDF([sig_ext, bkg_ext])

    # Fit yields
    data = zfit.Data.from_numpy(obs=B_mass, array=mass_mix)
    loss = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss)
    result.hesse()

    print("Combined yields fit: converged=", result.converged, " valid=", getattr(result, "valid", None))
    print("Nsig:", float(Nsig.value()), "Nbkg:", float(Nbkg.value()))


    # -------------------------
    # Configure observable space
    # -------------------------
    B_mass = zfit.Space("B_mass", limits=(5.170, 5.500))

    # Outputs
    os.makedirs("outputs/massfit", exist_ok=True)
    os.makedirs("outputs/sweights_checks", exist_ok=True)

    # -------------------------
    # Fit
    # -------------------------

    save_fitresult(result, "outputs/massfit/mass_fit_result.json")
    plot_mass_fit(mass_mix, B_mass, model, sig_ext, bkg_ext, "outputs/massfit/mass_fit.png")

    # -------------------------
    # sWeights
    # -------------------------
    print("\nFIT STATUS")
    print("  converged:", result.converged)
    print("  valid:    ", getattr(result, "valid", "N/A"))

    print("\nFITTED PARAMS")
    print("  Nsig:", float(Nsig.value()))
    print("  Nbkg:", float(Nbkg.value()))
    print("  lambda_bkg:", float(lambda_bkg.value()) if "lambda_bkg" in locals() else "lambda param not named lambda_bkg")

    print("\nMODEL CHECKS")
    print("  model.is_extended:", model.is_extended)
    print("  components extended:", [pdf.is_extended for pdf in model.pdfs])
    sweights = compute_sweights(model, data)
    print("sweights type:", type(sweights))
    try:
        print("sweights keys:", list(sweights.keys()))
    except Exception as e:
        print("could not list keys:", e)
        print("sweights repr:", sweights)
    w_sig = np.asarray(sweights[Nsig])
    w_bkg = np.asarray(sweights[Nbkg])

    assert len(mass_mix) == len(w_sig) == len(w_bkg) == len(angles_mix["cosThetaK"]) == len(angles_mix["cosThetaL"])

    # -------------------------
    # Save combined events + sWeights to ROOT (for angular fit)
    # -------------------------

    os.makedirs("outputs", exist_ok=True)
    out_root = "outputs/combined_with_sweights.root"

    # Safety check: all arrays must align 1-to-1 by event
    assert len(mass_mix) == len(w_sig) == len(w_bkg) == len(angles_mix["cosThetaK"]) == len(angles_mix["cosThetaL"])

    with uproot.recreate(out_root) as f:
        f["events"] = {
            "B_mass": mass_mix.astype("float64"),
            "cosThetaK": angles_mix["cosThetaK"].astype("float64"),
            "cosThetaL": angles_mix["cosThetaL"].astype("float64"),
            "w_sig": w_sig.astype("float64"),
            "w_bkg": w_bkg.astype("float64"),
        }

    print(f"Wrote {out_root} (tree: events)")


    print("sum(w_sig) =", float(np.sum(w_sig)), "  fitted Nsig =", float(Nsig.value()))
    print("sum(w_bkg) =", float(np.sum(w_bkg)), "  fitted Nbkg =", float(Nbkg.value()))


    # Basic sanity check
    print("Fit Nsig:", float(zfit.run(Nsig)), " Sum(w_sig):", float(np.sum(w_sig)))
    print("Fit Nbkg:", float(zfit.run(Nbkg)), " Sum(w_bkg):", float(np.sum(w_bkg)))

    # -------------------------
    # Per-angle check plots
    # -------------------------
    angle_names = ["cosThetaK", "cosThetaL"]
    plot_sweight_checks_per_angle(
        angles_mix=angles_mix,
        angles_sig_true=angles_sig_true,
        angles_bkg_true=angles_bkg_true,
        w_sig=w_sig,
        w_bkg=w_bkg,
        angle_names=angle_names,
        out_dir="outputs/sweights_checks",
    )

    print("Done. Check outputs/massfit and outputs/sweights_checks")


if __name__ == "__main__":
    main()
