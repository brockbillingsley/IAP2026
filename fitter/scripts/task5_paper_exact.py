import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

H5 = "sweights/standard/data_qsq-1.1-7.0/0.h5"
OUTDIR = "plots_compare_paper_exact"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_hdf(H5)

needed = ["mKpi", "q2", "wS", "wA0", "wApp", "wAq"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in H5: {missing}. Found: {df.columns.tolist()}")

# q2 window for this run
q2min, q2max = 1.1, 7.0

# Paper binning
mk_edges = np.arange(0.65, 1.50 + 0.02, 0.02)   # 0.02 GeV/c^2
q2_edges = np.arange(q2min, q2max + 0.12, 0.12) # 0.12 GeV^2/c^4

def sumw_hist(x, w, edges):
    """Sum of weights per bin, and sqrt(sum w^2) uncertainty."""
    sumw, _ = np.histogram(x, bins=edges, weights=w)
    sumw2, _ = np.histogram(x, bins=edges, weights=w*w)
    err = np.sqrt(sumw2)
    widths = edges[1:] - edges[:-1]
    centers = 0.5*(edges[1:] + edges[:-1])
    return centers, sumw/widths, err/widths, widths

def paper_stack(var, edges, title, out_png):
    x = df[var].to_numpy()

    # Component definitions (order matters for stacking visuals)
    comps = [
        ("wS",   "n_S  (wS)"),        # yellow-ish in paper; we won't control colors
        ("wA0",  "n_P^0 (wA0)"),
        ("wApp", "n_P^1 (wApp/A1)"),
        ("wAq",  "n_beta (wAq)"),
    ]

    # Compute component bin heights
    centers = None
    widths = None
    ys = []
    es2 = []  # for toy-data error estimate
    for wcol, _ in comps:
        c, y, e, widths = sumw_hist(x, df[wcol].to_numpy(), edges)
        centers = c
        ys.append(y)
        es2.append(e*e)

    ys = np.vstack(ys)  # shape (ncomp, nbins)

    # Toy data points: total = sum of stacked components in each bin
    ytot = np.sum(ys, axis=0)
    # Errorbars: rough estimate sqrt(sum of component variances)
    etot = np.sqrt(np.sum(es2, axis=0))

    plt.figure()

    # Draw stacked filled blocks without interior outline lines:
    baseline = np.zeros_like(ytot)
    for (wcol, label), y in zip(comps, ys):
        top = baseline + y
        # Use fill_between with step='post' to mimic histogram blocks
        plt.fill_between(edges[:-1], baseline, top, step="post", alpha=0.55, label=label)
        baseline = top

    # Overlay toy data points on top (black in paper; we won’t force color)
    plt.errorbar(centers, ytot, yerr=etot, fmt="o", markersize=3, label="Toy data (sum weights)")

    plt.title(title)
    plt.xlabel(var)
    plt.ylabel(r"$\sum$ weights / bin width")
    plt.legend(fontsize=8, loc="best")

    # Set y-limit based on the total (avoid spikes ruining view)
    ymax = 1.15 * float(np.nanpercentile(ytot, 99.5))
    if ymax > 0:
        plt.ylim(0, ymax)

    plt.savefig(os.path.join(OUTDIR, out_png), dpi=200, bbox_inches="tight")
    plt.close()

paper_stack("mKpi", mk_edges, "mKpi: stacked weighted components (paper-exact style)", "mKpi_stack.png")
paper_stack("q2",   q2_edges, "q2: stacked weighted components (paper-exact style)",  "q2_stack.png")

print("Wrote:", os.path.join(OUTDIR, "mKpi_stack.png"))
print("Wrote:", os.path.join(OUTDIR, "q2_stack.png"))
