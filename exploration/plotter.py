import argparse
import matplotlib.pyplot as plt
import uproot

parser = argparse.ArgumentParser(description="Plot histograms from ROOT file.")
parser.add_argument("--data", type=str, required=True, help="Path to the input ROOT file.")
args = parser.parse_args()

# Open the ROOT file
with uproot.open(args.data) as file:
    data = file["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")

# ----- Task 1: Data Inspection -----
print("\n--- TASK 1: BASIC INSPECTION ---")
print("Rows:", len(data))
print("Columns:", len(data.columns))
print("\nColumn names:", list(data.columns))
print("\nFirst 5 rows:\n", data.head())
print("\nDtypes:\n", data.dtypes)
print("\nSummary stats:\n", data.describe())
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

# ----- Task 5: 2D histograms -----
import os
import numpy as np

os.makedirs("plots", exist_ok=True)

# 1/11 million sampled to speed up
DO_SAMPLE = True
N_SAMPLE = 1_000_000

df2d = data
if DO_SAMPLE and len(data) > N_SAMPLE:
    df2d = data.sample(n=N_SAMPLE, random_state=0)

# 5.1) 2D histogram: cosThetaK vs cosThetaL
x = df2d["cosThetaK"].to_numpy()
y = df2d["cosThetaL"].to_numpy()

plt.figure(figsize=(7, 6))
plt.hist2d(
    x, y,
    bins=80,
    range=[[-1, 1], [-1, 1]],
    density=True
)
plt.colorbar(label="Normalized density")
plt.xlabel("cosThetaK (dimensionless)")
plt.ylabel("cosThetaL (dimensionless)")
plt.title("2D histogram: cosThetaK vs cosThetaL")
plt.tight_layout()
plt.savefig("plots/cosThetaK_vs_cosThetaL_2D.png", dpi=200)
plt.close()

# 5.2) 2D histogram: mKpi vs q2
x = df2d["mKpi"].to_numpy()
y = df2d["q2"].to_numpy()

plt.figure(figsize=(7, 6))
plt.hist2d(
    x, y,
    bins=[80, 80],
    range=[[0.6, 1.8], [0, 12.5]],
    density=True
)
plt.colorbar(label="Normalized density")
plt.xlabel("mKpi (GeV)")
plt.ylabel("q2 (GeV$^2$)")
plt.title("2D histogram: mKpi vs q2")
plt.tight_layout()
plt.savefig("plots/mKpi_vs_q2_2D.png", dpi=200)
plt.close()

print("Task 5 done: wrote 2D histograms to plots/")

# Task 1) Inspect the data and identify the different variables. What are their units?
# Task 2) Create one plot per variable with five different distributions:
#           a) the total distribution
#           b,c) the distribution when selecting q2 smaller or larger than 2
#           d,e) the distribution when selecting mKpi smaller or larger than 1.1
#         Each distribution should be represented by a histogram
#         Save the plots as png or pdf files.
# Task 3) Do your figures have all the required features? (E.g. axis labels, legend, easily distinguishable colors or linestyles, etc.)
# Task 4) Can you interpret the differences between the distributions a-e?
# Task 5) Create a 2D histogram of cosThetaK and cosThetaL. Create another 2D histogram of mKpi and q2.
#         Save both histograms as png or pdf files.
