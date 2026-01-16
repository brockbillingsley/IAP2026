import uproot
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fname = "bkg_toy.root"   # <-- CHANGE to your actual output file name

f = uproot.open(fname)
print("Keys in ROOT file:", f.keys())

tree = f[f.keys()[0]]  # <-- if this isn't the tree, we'll adjust after seeing keys
print("Using tree:", tree.name)
print("Branches:", tree.keys())

arr = tree.arrays(["cosThetaK", "cosThetaL"], library="np")
ctk = arr["cosThetaK"]
ctl = arr["cosThetaL"]

print("cosThetaK min/max:", ctk.min(), ctk.max())
print("cosThetaL min/max:", ctl.min(), ctl.max())

plt.figure()
plt.hist(ctk, bins=50, range=(-1, 1))
plt.xlabel("cosThetaK")
plt.ylabel("Events")
plt.tight_layout()
plt.savefig("cosThetaK_quickcheck.png", dpi=200)
plt.close()

plt.figure()
plt.hist(ctl, bins=50, range=(-1, 1))
plt.xlabel("cosThetaL")
plt.ylabel("Events")
plt.tight_layout()
plt.savefig("cosThetaL_quickcheck.png", dpi=200)
plt.close()

print("Saved: cosThetaK_quickcheck.png, cosThetaL_quickcheck.png")
