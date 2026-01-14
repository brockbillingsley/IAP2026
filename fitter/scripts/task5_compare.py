import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

H5 = "sweights/standard/data_qsq-1.1-7.0/0.h5"
DATADIR = os.environ.get("DATADIR")
assert DATADIR, "DATADIR env var not set. Run: export DATADIR=/path/to/data"

# --- load weighted data ---
df = pd.read_hdf(H5)

# check required columns
needed = ["mKpi", "q2", "wA0", "wApp", "wS", "wAq"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in {H5}: {missing}. Found: {df.columns.tolist()}")

# --- load reference samples (only A0 and AS exist for you) ---
def load_root_df(path):
    f = uproot.open(path)
    # take first tree-like object
    tree = None
    for _, obj in f.items():
        if hasattr(obj, "arrays"):
            tree = obj
            break
    if tree is None:
        raise RuntimeError(f"No TTree found in {path}. Keys: {list(f.keys())}")
    return tree.arrays(["mKpi", "q2"], library="pd")

ref = {}
a0_path = os.path.join(DATADIR, "A0.root")
as_path = os.path.join(DATADIR, "AS.root")

if os.path.exists(a0_path):
    ref["A0"] = load_root_df(a0_path)
else:
    print("Warning: A0.root not found at", a0_path)

if os.path.exists(as_path):
    ref["AS"] = load_root_df(as_path)
else:
    print("Warning: AS.root not found at", as_path)

# --- histogram overlay helper ---
def overlay(var, weight_col, ref_df, title, out_png, bins=60):
    x = df[var].to_numpy()
    w = df[weight_col].to_numpy()

    # robust plotting range from weighted data
    xmin = float(np.nanpercentile(x, 0.5))
    xmax = float(np.nanpercentile(x, 99.5))

    plt.figure()

    # weighted data (normalize for shape comparison)
    hw, edges = np.histogram(x, bins=bins, range=(xmin, xmax), weights=w)
    centers = 0.5 * (edges[1:] + edges[:-1])
    hw_sum = hw.sum() if hw.sum() != 0 else 1.0
    plt.step(centers, hw / hw_sum, where="mid", label=f"data weighted ({weight_col})")

    # reference (unweighted)
    if ref_df is not None:
        xr = ref_df[var].to_numpy()
        hr, edgesr = np.histogram(xr, bins=bins, range=(xmin, xmax))
        centersr = 0.5 * (edgesr[1:] + edgesr[:-1])
        hr_sum = hr.sum() if hr.sum() != 0 else 1.0
        plt.step(centersr, hr / hr_sum, where="mid", label="reference")

    plt.xlabel(var)
    plt.ylabel("normalized counts")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join("plots_compare", out_png), dpi=200, bbox_inches="tight")
    plt.close()

# --- make plots ---
# A0: compare weighted vs A0.root
if "A0" in ref:
    overlay("mKpi", "wA0", ref["A0"], "A0: mKpi (weighted vs reference)", "A0_mKpi.png")
    overlay("q2",   "wA0", ref["A0"], "A0: q2 (weighted vs reference)",   "A0_q2.png")

# AS: code weight column is wS, reference file is AS.root
if "AS" in ref:
    overlay("mKpi", "wS", ref["AS"], "AS: mKpi (weighted vs reference)", "AS_mKpi.png")
    overlay("q2",   "wS", ref["AS"], "AS: q2 (weighted vs reference)",   "AS_q2.png")

# App: weighted-only (no reference file found)
overlay("mKpi", "wApp", None, "App: mKpi (weighted, no reference file)", "App_mKpi.png")
overlay("q2",   "wApp", None, "App: q2 (weighted, no reference file)",   "App_q2.png")

# Aq/nbeta: weighted-only (assignment says no reference)
overlay("mKpi", "wAq", None, "Aq/nbeta: mKpi (weighted, no reference)", "Aq_mKpi.png")
overlay("q2",   "wAq", None, "Aq/nbeta: q2 (weighted, no reference)",   "Aq_q2.png")

print("Done. Saved plots in plots_compare/")
