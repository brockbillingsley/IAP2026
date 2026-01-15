import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep
from hist import Hist
import hist
import uproot


mplhep.style.use("CMS")

H5 = "sweights/standard/data_qsq-1.1-7.0/0.h5"
OUTDIR = "plots_compare_fitter_style"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_hdf(H5)

DATADIR = os.environ.get("DATADIR")
assert DATADIR, "DATADIR not set. Run: export DATADIR=/path/to/data"

def load_ref_array(root_path, branches=("mKpi", "q2")):
    f = uproot.open(root_path)
    tree = None
    for _, obj in f.items():
        if hasattr(obj, "arrays"):
            tree = obj
            break
    if tree is None:
        raise RuntimeError(f"No TTree found in {root_path}. Keys: {list(f.keys())}")
    # load as numpy dict
    arr = tree.arrays(list(branches), library="np")
    return arr

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
    if nbins < 5:
        nbins = 5

    # Toy data (unweighted counts)
    H = Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
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

    lists = zip(
        [
            r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$",
            r"$n_0^P=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$",
            r"$n_1^P=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$",
            r"$n_{\beta}$",
        ],
        [sAS, sA0, sApp, sAq],
        ["gold", "navy", "dodgerblue", "firebrick"],
    )

    for n, w, c in lists:
        if np.all(w == 0):
            continue

        # Weighted histogram with proper variances
        Hw = Hist(
            hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False),
            storage=hist.storage.Weight(),
        )
        Hw.fill(datatoy[vkey].to_numpy(), weight=w)

        hvals = Hw.values()         # sum of weights per bin
        hvars = Hw.variances()      # variance per bin

        # Fill stacked bands bin-by-bin (no edge lines)
        edges = Hw.axes[0].edges
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
            label=n,
        )

        y += hvals  # raise baseline

    # Reference overlays (scaled)
    for rkey, info in ref_map.items():
        if rkey not in refs:
            continue

        # Reference variable array
        xref = refs[rkey][vkey]

        # Apply same plotting range as data
        xref = xref[(xref >= mi) & (xref <= ma)]
        if len(xref) == 0:
            continue

        # Reference histogram in SAME bins (counts per bin)
        Href = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
        Href.fill(xref)
        ref_counts = Href.values()

        # Convert to density-like scale: counts / binwidth (to match ylabel form)
        binwidth = dist / nbins

        # Compute corresponding component yield from weights in the same bins
    # Reference overlays (scaled), in SAME units as stack: counts/weights per bin
    # Save y-lims so references can't autoscale and crush the stack
    ylims_before_refs = plt.ylim()

    for rkey, info in ref_map.items():
        if rkey not in refs:
            continue

        xref = refs[rkey][vkey]
        xref = xref[(xref >= mi) & (xref <= ma)]
        if len(xref) == 0:
            continue

        # Reference histogram: COUNTS PER BIN (no /binwidth)
        Href = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
        Href.fill(xref)
        ref_counts = Href.values()  # counts/bin

        # Component yield in SAME bins: SUM OF WEIGHTS PER BIN (no /binwidth)
        wcomp = datatoy[info["wcol"]].to_numpy()
        Hcomp = hist.Hist(
            hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False),
            storage=hist.storage.Weight(),
        )
        Hcomp.fill(datatoy[vkey].to_numpy(), weight=wcomp)
        comp_w = Hcomp.values()  # sumw/bin

        # Scale reference so its integral matches the component integral
        ref_int = float(np.sum(ref_counts))
        comp_int = float(np.sum(comp_w))
        scale = (comp_int / ref_int) if ref_int > 0 else 1.0

        edges = Href.axes[0].edges
        y_step = np.r_[ref_counts * scale, (ref_counts * scale)[-1]]

        plt.step(
            edges,
            y_step,
            where="post",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label=f"Ref {rkey} (scaled)",
            zorder=5,   # above fills, below points is fine
        )

    # Restore y-lims so refs do NOT rescale the plot
    plt.ylim(ylims_before_refs)



    plt.legend(handletextpad=0.1, fontsize=10, ncol=1)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlim(mi, ma)

    plt.xlabel(l + f" [{u}]", ha="right", x=1)
    plt.ylabel(fr"$\sum$ weights / ({dist/nbins:.2f} {u})", ha="right", y=1)

    outpng = os.path.join(OUTDIR, f"{vkey}_fitter_style.png")
    plt.savefig(outpng, dpi=200)
    plt.close()

    print("Wrote:", outpng)
