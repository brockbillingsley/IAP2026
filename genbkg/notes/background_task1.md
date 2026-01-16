Task 1 — Background toy generation + generator shapes

I generated a background toy sample with `genbkg/generator.py`, which builds a 5D background PDF:

Parameters are read from `background_params.json`. The 1D shapes used are:

- **q2:** Uniform on \([1.1, 7.0]\)
- **mKpi:** Uniform on \([0.65, 1.5]\)
- **B_mass:** Exponential on \([5.170, 5.500]\)
- **cosThetaK:** Legendre expansion on \([-1,1]\) with coefficients \([a1\_cosh, a2\_cosh]\)
- **cosThetaL:** Legendre expansion on \([-1,1]\) with coefficients \([a1\_cosl, a2\_cosl]\)

The toy is saved to a ROOT file via `uproot` as a TTree named `background` with branches:
`q2`, `mKpi`, `B_mass`, `cosThetaK`, `cosThetaL`. I verified the sample by plotting `cosThetaK` and `cosThetaL` and confirmed they match the expected Legendre-type angular shapes.

Quick check plots
![cosThetaK](../genbkg/plots/cosThetaK_quickcheck.png)
![cosThetaL](../genbkg/plots/cosThetaL_quickcheck.png)
