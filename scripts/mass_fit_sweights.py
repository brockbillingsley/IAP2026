import os
import json
import numpy as np
import matplotlib.pyplot as plt

import zfit
znp = zfit.z.numpy

# Try core zfit first; otherwise use zfit_physics
try:
    GeneralizedCB = zfit.pdf.GeneralizedCB
except AttributeError:
    from zfit_physics.pdf import GeneralizedCB

from hepstats.splot import compute_sweights

def fit_shape_unbinned(pdf, mass_array, obs, label="shape"):
    data = zfit.Data.from_numpy(obs=obs, array=np.asarray(mass_array, dtype=np.float64))
    loss = zfit.loss.UnbinnedNLL(model=pdf, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss)
    result.hesse()
    print(f"{label} fit: converged={result.converged} valid={getattr(result, 'valid', None)}")
    print(result)
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
    data = zfit.Data.from_numpy(obs=B_mass_obs, array=np.asarray(mass_array, dtype=np.float64))
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

def plot_mass_fit(data_np, B_mass_obs, model, sig_ext, bkg_ext, outpath, nbins=60):
    # Range from observable
    lo, hi = 5.170, 5.500

    # Bin the data
    counts, edges = np.histogram(data_np, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    # Errors
    yerr = np.sqrt(counts)

    # Evaluate extended PDFs at bin centers
    x = zfit.z.convert_to_tensor(centers)
    w = zfit.z.convert_to_tensor(widths)


    # ext_pdf is events per unit x -> multiply by bin width -> counts per bin
    y_tot = znp.asarray(model.ext_pdf(x) * w)
    y_sig = znp.asarray(sig_ext.ext_pdf(x) * w)
    y_bkg = znp.asarray(bkg_ext.ext_pdf(x) * w)

    # Plot
    plt.figure(figsize=(9, 6))

    plt.errorbar(
        centers, counts, yerr=yerr,
        fmt="o", markersize=3,
        capsize=2, elinewidth=1.2,
        label="data",
        zorder=10,   # draw on top
    )


    plt.plot(centers, y_tot, label="total", linewidth=2, zorder=2)
    plt.plot(centers, y_sig, label="signal", linewidth=2, zorder=2)
    plt.plot(centers, y_bkg, label="background", linewidth=2, zorder=2)


    plt.xlabel("B_mass")
    plt.ylabel("Counts / bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# 4) Per-angle sWeight checks
# -----------------------------
def plot_sweight_overlays_per_angle(
    angles_mix, angles_sig_true, angles_bkg_true,
    w_sig, w_bkg,
    angle_names,
    out_dir,
    nbins=40,
    density=True,   # True = compare shapes; False = compare raw yields
):
    os.makedirs(out_dir, exist_ok=True)

    for ang in angle_names:
        x_mix = np.asarray(angles_mix[ang])
        x_sig = np.asarray(angles_sig_true[ang])
        x_bkg = np.asarray(angles_bkg_true[ang])

        # Binning
        lo = float(np.nanmin(np.concatenate([x_mix, x_sig, x_bkg])))
        hi = float(np.nanmax(np.concatenate([x_mix, x_sig, x_bkg])))
        bins = np.linspace(lo, hi, nbins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        widths = np.diff(bins)

        # SIGNAL overlay: (combined×w_sig) vs (truth sig) ---
        hw_sig, _ = np.histogram(x_mix, bins=bins, weights=w_sig)
        ht_sig, _ = np.histogram(x_sig, bins=bins)

        if density:
            area_hw = np.sum(hw_sig * widths)
            area_ht = np.sum(ht_sig * widths)
            if area_hw > 0:
                hw_sig = hw_sig / area_hw
            if area_ht > 0:
                ht_sig = ht_sig / area_ht

        plt.figure(figsize=(7, 5))
        plt.step(centers, hw_sig, where="mid", linewidth=2, label="combined × signal sWeight")
        plt.step(centers, ht_sig, where="mid", linewidth=2, label="truth signal only")
        plt.xlabel(ang)
        plt.ylabel("Density" if density else "Entries")
        plt.title(f"{ang}: signal overlay")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"overlay_{ang}_signal.png"), dpi=200)
        plt.close()

        # --- BKG overlay: (combined×w_bkg) vs (truth bkg) ---
        hw_bkg, _ = np.histogram(x_mix, bins=bins, weights=w_bkg)
        ht_bkg, _ = np.histogram(x_bkg, bins=bins)

        if density:
            area_hw = np.sum(hw_bkg * widths)
            area_ht = np.sum(ht_bkg * widths)
            if area_hw > 0:
                hw_bkg = hw_bkg / area_hw
            if area_ht > 0:
                ht_bkg = ht_bkg / area_ht

        plt.figure(figsize=(7, 5))
        plt.step(centers, hw_bkg, where="mid", linewidth=2, label="combined × background sWeight")
        plt.step(centers, ht_bkg, where="mid", linewidth=2, label="truth background only")
        plt.xlabel(ang)
        plt.ylabel("Density" if density else "Entries")
        plt.title(f"{ang}: background overlay")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"overlay_{ang}_bkg.png"), dpi=200)
        plt.close()


# -----------------------------
# MAIN
# -----------------------------
def main():

    import uproot

    def load_root_sample(root_path, tree_name, branches):
        with uproot.open(root_path) as f:
            tree = f[tree_name]
            arr = tree.arrays(branches, library="pd")
        return arr

    # -------------------------
    # LOAD DATA HERE
    # -------------------------

    # -------------------------
    # Configure observable space
    # -------------------------
    B_mass = zfit.Space("B_mass", limits=(5.170, 5.500))


    sig_path = "/ceph/submit/data/user/a/anbeck/B2KPiMM_michele/full.root"
    bkg_path = "genbkg/bkg_toy.root"

    sig_tree = "B02KstMuMu_Run1_centralQ2E_sig"
    bkg_tree = "background"

    branches = ["B_mass", "cosThetaK", "cosThetaL", "mKpi", "q2"]

    arr_sig = load_root_sample(sig_path, sig_tree, branches)
    arr_bkg = load_root_sample(bkg_path, bkg_tree, branches)

    arr_sig["B_mass"] /= 1000
    arr_sig = arr_sig.query(f"(0.65<mKpi) &(mKpi<1.5)")
    arr_sig = arr_sig.query(f"(1.1<q2) &(q2<7)")
    arr_sig = arr_sig.query(f"(5.17<B_mass) &(B_mass<5.5)")

    arr_bkg = arr_bkg.query(f"(0.65<mKpi) &(mKpi<1.5)")
    arr_bkg = arr_bkg.query(f"(1.1<q2) &(q2<7)")
    arr_bkg = arr_bkg.query(f"(5.17<B_mass) &(B_mass<5.5)")


    mass_sig = arr_sig["B_mass"].to_numpy(dtype=np.float64)
    mass_bkg = arr_bkg["B_mass"].to_numpy(dtype=np.float64)


    angles_sig_true = {
        "cosThetaK": arr_sig["cosThetaK"].to_numpy(dtype=np.float64),
        "cosThetaL": arr_sig["cosThetaL"].to_numpy(dtype=np.float64),
    }
    angles_bkg_true = {
        "cosThetaK": arr_bkg["cosThetaK"].to_numpy(dtype=np.float64),
        "cosThetaL": arr_bkg["cosThetaL"].to_numpy(dtype=np.float64),
    }

    # Keep mKpi and q2 arrays
    mkpi_sig = arr_sig["mKpi"].to_numpy() if hasattr(arr_sig["mKpi"], "to_numpy") else np.asarray(arr_sig["mKpi"])
    mkpi_bkg = arr_bkg["mKpi"].to_numpy() if hasattr(arr_bkg["mKpi"], "to_numpy") else np.asarray(arr_bkg["mKpi"])

    q2_sig = arr_sig["q2"].to_numpy() if hasattr(arr_sig["q2"], "to_numpy") else np.asarray(arr_sig["q2"])
    q2_bkg = arr_bkg["q2"].to_numpy() if hasattr(arr_bkg["q2"], "to_numpy") else np.asarray(arr_bkg["q2"])



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
    print("DEBUG mix lens:", len(mass_sig), len(mass_bkg), len(mass_mix))
    assert len(mass_mix) == len(mass_sig) + len(mass_bkg)

    angles_mix = {
        "cosThetaK": np.concatenate([angles_sig_true["cosThetaK"], angles_bkg_true["cosThetaK"]]),
        "cosThetaL": np.concatenate([angles_sig_true["cosThetaL"], angles_bkg_true["cosThetaL"]]),
    }
    assert len(angles_mix["cosThetaK"]) == len(mass_mix)
    assert len(angles_mix["cosThetaL"]) == len(mass_mix)

    mkpi_mix = np.concatenate([mkpi_sig, mkpi_bkg])
    q2_mix   = np.concatenate([q2_sig, q2_bkg])

    assert len(mkpi_mix) == len(mass_mix)
    assert len(q2_mix) == len(mass_mix)


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
    assert len(mass_mix) == len(angles_mix["cosThetaK"]) == len(angles_mix["cosThetaL"]) == len(mkpi_mix) == len(q2_mix) == len(w_sig) == len(w_bkg)

    with uproot.recreate(out_root) as f:
        f["events"] = {
            "B_mass":    np.asarray(mass_mix, dtype="float64"),
            "cosThetaK": np.asarray(angles_mix["cosThetaK"], dtype="float64"),
            "cosThetaL": np.asarray(angles_mix["cosThetaL"], dtype="float64"),
            "mKpi":      np.asarray(mkpi_mix, dtype="float64"),
            "q2":        np.asarray(q2_mix, dtype="float64"),
            "w_sig":     np.asarray(w_sig, dtype="float64"),
            "w_bkg":     np.asarray(w_bkg, dtype="float64"),
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
    plot_sweight_overlays_per_angle(
        angles_mix=angles_mix,
        angles_sig_true=angles_sig_true,
        angles_bkg_true=angles_bkg_true,
        w_sig=w_sig,
        w_bkg=w_bkg,
        angle_names=["cosThetaK", "cosThetaL"],
        out_dir="outputs/sweights_checks",
        nbins=40,
        density=True,
    )

    print("Done. Check outputs/massfit and outputs/sweights_checks")


if __name__ == "__main__":
    main()
