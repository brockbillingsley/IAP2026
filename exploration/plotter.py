import argparse
import matplotlib.pyplot as plt
import uproot

parser = argparse.ArgumentParser(description="Plot histograms from ROOT file.")
parser.add_argument("--data", type=str, help="Path to the input ROOT file.")
args = parser.parse_args()

# Open the ROOT file
with uproot.open(args.data) as file:
    data = file["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")

# ----- Task 1: Data Inspection -----
print("\nFor task 1, please also see the write-up in task1.md")

    print("\n=== TASK 1: BASIC INSPECTION ===")
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    print("\nColumn names:")
    print(list(data.columns))

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nDtypes:")
    print(data.dtypes)

    print("\nSummary stats:")
    print(data.describe())

import os
import numpy as np

# ----- Task 2: plotting -----
outdir = "plots"
os.makedirs(outdir, exist_ok=True)

# Define the 5 selections (a-e)
sel_total = np.ones(len(data), dtype=bool)          # a) all events
sel_q2_low = data["q2"] < 2                         # b) q2 < 2
sel_q2_high = data["q2"] > 2                        # c) q2 > 2
sel_mkpi_low = data["mKpi"] < 1.1                   # d) mKpi < 1.1
sel_mkpi_high = data["mKpi"] > 1.1                  # e) mKpi > 1.1

selections = [
    ("total", sel_total),
    ("q2 < 2", sel_q2_low),
    ("q2 > 2", sel_q2_high),
    ("mKpi < 1.1", sel_mkpi_low),
    ("mKpi > 1.1", sel_mkpi_high),
]

# The six variables to plot (one plot each)
variables = ["cosThetaK", "cosThetaL", "phi", "q2", "B_mass", "mKpi"]

for var in variables:
    plt.figure()

    for label, sel in selections:
        values = data.loc[sel, var].dropna().to_numpy()

        plt.hist(
            values,
            bins=80,
            histtype="step",
            linewidth=1.5,
            density=True,
            label=label,
        )

    # Axis labels (including units where relevant)
    if var in ["cosThetaK", "cosThetaL"]:
        plt.xlabel(var + " (dimensionless)")
    elif var == "phi":
        plt.xlabel(var + " (rad)")
    elif var == "q2":
        plt.xlabel(var + " (GeV$^2$)")
    elif var == "B_mass":
        plt.xlabel(var + " (MeV)")
    elif var == "mKpi":
        plt.xlabel(var + " (GeV)")
    else:
        plt.xlabel(var)

    plt.ylabel("Normalized entries")
    plt.title(f"{var}: total vs selections")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{var}.png"), dpi=200)
    plt.close()

print(f"\nSaved plots to: {outdir}/")

# ----- Task 3: Required plot features -----
print("\nYes, each plot contains all necessary elements.")

# ----- Task 4: Interpret the differences between distributions (a-e) -----
print("\nTask 4 write-up contained in task4.md.")

# Task 1) Inspect the data and identify the different variables. What are their units?
# Task 2) Create one plot per variable with five different distributions, each represented as a histogram:
#           a) the total distribution
#           b,c) the distribution when selecting q2 smaller or larger than 2
#           d,e) the distribution when selecting mKpi smaller or larger than 1.1
# Task 3) Do your figures have all the required features? (E.g. axis labels, legend, easily distinguishable colors or linestyles, etc.)
# Task 4) Can you interpret the differences between the distributions a-e?
