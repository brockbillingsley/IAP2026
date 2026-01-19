import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep
import hist
import hist
import uproot


mplhep.style.use("CMS")

H5 = "fitter/sweights/standard/data_qsq-1.1-7.0/0.h5"
OUTDIR = "plots_compare_fitter_style"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_hdf(H5)

DATADIR = os.environ.get("DATADIR")
assert DATADIR, "DATADIR not set. Run: export DATADIR=/path/to/data"

def load_ref_array(root_path, branches=("mKpi", "q2")):
    with uproot.open(root_path) as f:
        # Pick the first TTree that contains ALL requested branches
        for k, obj in f.items():
            try:
                # uproot TTrees have .keys() listing branch names
                if hasattr(obj, "keys") and all(b in obj.keys() for b in branches):
                    return obj.arrays(list(branches), library="np")
            except Exception:
                continue

        # If none matched, print helpful debugging info
        keys = list(f.keys())
        raise RuntimeError(
            f"No TTree with branches {branches} found in {root_path}. "
            f"Top-level keys: {keys}"
        )

# Map component -> reference file + which variable column it corresponds to in weights
ref_map = {
    "A0": {"file": os.path.join(DATADIR, "A0.root"), "wcol": "wA0"},
    "AS": {"file": os.path.join(DATADIR, "AS.root"), "wcol": "wS"},
    "A1": {"file": os.path.join(DATADIR, "A1.root"), "wcol": "wApp"},  # App reference
}

refs = {}
for key, info in ref_map.items():
    if os.path.exists(info["file"]):
        refs[key] = load_ref_array(info["file"])
print("Reference overlays available:", list(refs.keys()))


# Required columns based on Task 4 output
for col in ["mKpi", "q2", "wS", "wA0", "wApp", "wAq"]:
    if col not in df.columns:
        raise KeyError(f"Missing {col} in {H5}. Found: {df.columns.tolist()}")

datatoy = df

# weights
sAS  = datatoy["wS"].to_numpy()
sA0  = datatoy["wA0"].to_numpy()
sApp = datatoy["wApp"].to_numpy()
sAq  = datatoy["wAq"].to_numpy()

# mKpi: ~0.02; q2: ~0.12
binwidths = {"mKpi": 0.02, "q2": 0.12}

for vkey, l, u in [
    ("mKpi", r"$m(K\pi)$", r"GeV$/c^2$"),
    ("q2",   r"$q^2$",     r"GeV$^2/c^4$"),

]:
    mi, ma = float(datatoy[vkey].min()), float(datatoy[vkey].max())
    dist = ma - mi
    bw = binwidths[vkey]
    nbins = int(np.floor(dist / bw))
    print("DEBUG: vkey =", vkey)

    if nbins < 5:
        nbins = 5

    # Toy data (unweighted counts)
    H = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
    H.fill(datatoy[vkey].to_numpy())

    plt.figure()
    mplhep.histplot(
        H,
        color="black",
        histtype="errorbar",
        label="Toy data",
        xerr=True,
        yerr=True,
        marker=".",
        zorder=20,
    )

    y = np.zeros(nbins)  # stacking baseline
    comp_cache = {}  # store component histograms so refs can match exactly

    lists = [
    ("AS",  r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$", sAS,  "gold"),
    ("A0",  r"$n_0^P=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$",   sA0,  "navy"),
    ("A1",  r"$n_1^P=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$", sApp, "dodgerblue"),
    ("Aq",  r"$n_{\beta}$",                              sAq,  "firebrick"),
]

    for key, label, w, c in lists:
        if np.all(w == 0):
            continue

        # Weighted histogram with proper variances
        Hw = hist.Hist(
            hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False),
            storage=hist.storage.Weight(),
        )
        Hw.fill(datatoy[vkey].to_numpy(), weight=w)

        hvals = Hw.values()         # sum of weights per bin

        hvars = Hw.variances()      # variance per bin

        # Fill stacked bands bin-by-bin (no edge lines)
        edges = Hw.axes[0].edges

        # Cache component histograms so we can overlay an exact-outline reference
        # identify components by which weight column they came from.
        if key == "AS":
            comp_cache["AS"] = (edges.copy(), hvals.copy())

        centers = Hw.axes[0].centers
        widths = Hw.axes[0].widths

        for k in range(nbins):
            plt.fill_between(
                edges[k:k+2],
                y1=y[k],
                y2=y[k] + hvals[k],
                color=c,
                linewidth=0,
                zorder=0,
            )

        # Component errorbars on top of cumulative stack
        plt.errorbar(
            centers,
            y + hvals,
            yerr=np.sqrt(hvars),
            xerr=widths / 2,
            fmt=".",
            elinewidth=1,
            color=c,
            label=label,
        )

        y += hvals  # raise baseline

    # Save y-lims so references can't autoscale and crush the stack
    ylims_before_refs = plt.ylim()

        # EXACT outline reference for AS (should match yellow fill bin-by-bin)
    if "AS" in comp_cache:
        edges_as, hvals_as = comp_cache["AS"]
        y_step_as = np.r_[hvals_as, hvals_as[-1]]
        plt.step(
            edges_as,
            y_step_as,
            where="post",
            linestyle="--",
            linewidth=1.6,
            alpha=0.95,
            label="Ref AS (exact)",
            zorder=6,
        )
    else:
        print(f"WARNING: AS not cached for {vkey} (no exact outline drawn)")

    # keep external references for the others
    for rkey, info in ref_map.items():
        if rkey == "AS":
            continue  # we already drew exact AS
        if rkey not in refs:
            continue

        # Apply the SAME selection window as the H5 sample to the reference sample
        mKpi_min, mKpi_max = float(datatoy["mKpi"].min()), float(datatoy["mKpi"].max())
        q2_min,   q2_max   = float(datatoy["q2"].min()),   float(datatoy["q2"].max())

        ref_mKpi = refs[rkey]["mKpi"]
        ref_q2   = refs[rkey]["q2"]

        # Always require both variables to be inside the same windows as the toy sample
        mask = (
            (ref_mKpi >= mKpi_min) & (ref_mKpi <= mKpi_max) &
            (ref_q2   >= q2_min)   & (ref_q2   <= q2_max)
        )

        xref = refs[rkey][vkey][mask]

# Also enforce the plotting variable's range exactly (mi/ma are for current vkey)
        xref = xref[(xref >= mi) & (xref <= ma)]

        if len(xref) == 0:
            continue

        Href = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
        Href.fill(xref)
        ref_counts = Href.values()

        wcomp = datatoy[info["wcol"]].to_numpy()
        Hcomp = hist.Hist(
            hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False),
            storage=hist.storage.Weight(),
        )
        Hcomp.fill(datatoy[vkey].to_numpy(), weight=wcomp)
        comp_w = Hcomp.values()

        print("DEBUG", vkey, rkey,

            "Nref(after cuts)=", len(xref),
            "sum_ref_counts=", float(np.sum(ref_counts)),
            "sumw(comp)=", float(np.sum(comp_w)))


        ref_int = float(np.sum(ref_counts))
        comp_int = float(np.sum(comp_w))
        scale = (comp_int / ref_int) if ref_int > 0 else 1.0

        edges = Href.axes[0].edges
        y_step = np.r_[ref_counts * scale, (ref_counts * scale)[-1]]

        plt.step(edges, y_step, where="post", linestyle="--", linewidth=1.2,
                alpha=0.9, label=f"Ref {rkey} (scaled)", zorder=5)

    # Restore y-lims so refs do NOT rescale the plot
    plt.ylim(ylims_before_refs)

    # Final plot cosmetics + save (DO NOT indent this inside the ref_map loop)
    plt.legend(handletextpad=0.1, fontsize=10, ncol=1)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlim(mi, ma)

    plt.xlabel(l + f" [{u}]", ha="right", x=1)
    plt.ylabel(fr"$\sum$ weights / ({dist/nbins:.2f} {u})", ha="right", y=1)

    outpng = os.path.join(OUTDIR, f"{vkey}_fitter_style.png")
    plt.savefig(outpng, dpi=200)
    plt.close()

    print("Wrote:", outpng)
