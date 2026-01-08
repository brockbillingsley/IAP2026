# ----- Task 4: Interpret the differences between distributions (a-e) -----
task4_text = """\
# Task 4 - Interpretation of differences (a-e)

Definitions of the five distributions:
a) total = all events
b) q2 < 2
c) q2 > 2
d) mKpi < 1.1
e) mKpi > 1.1

## B_mass (MeV)
It appears that all five curves lie on top of eachother, sharing a narrow
peak near the B mass. Thus, we can conclude that B_mass is essentially
uncorrelated with the q2 and mKpi selections in thsi toy dataset.

## mKpi (GeV)
The total distribution shows a strong peak around the K* region (~0.9 GeV)
and a tail to higher masses.
- mKpi < 1.1 isolates the resonance region and thus produces a sharper peak.
- mKpi > 1.1 removes the resonance peak and gives a broad distribution above
1.1 GeV.
These q2 selections do not drastically reshape the K* peak compared the those
seen in the mKpi curves.

## q2 (GeV^2)
- q2 < 2 is confined to range [0,2] by definition
- q2 > 2 begins at 2 by definition and fills the higher-q2 region
The mKpi selections slightly change the q2 shape relative to that of the
total, showing mild correlation.

## cosThetaK (dimensionless)
The total distribution is U-shaped
mKpi > 1.1 looks flatter than the resonance-like selections, while mKpi < 1.1
tends to preserve the U-shape.
q2 < 2 also differs from q2 > 2, indicating dependence on the q2 region.

## cosThetaL (dimensionless)
The total distribution is a broad arch, with noticeable differences between
q2 < 2 q2 > 2.
mKpi > 1.1 also has a noticeably different shape from the resonance region.

## phi (rad)
The phi distribution is not flat, but rather shows wave-like structure.
q2 < 2 differs from q2 > 2, and mKpi > 1.1 deviates from the resonance-like
curves.

## Overall Interpretations
B_mass is stable under these selections, while the angular variables
(cosThetaK, cosThetaL, phi) change noticeably, especially when separating
the K* region (mKpi < 1.1) from the higher-mKpi region (mKpi > 1.1) and when
comparing low vs high q2 (q2 < 2 vs q2 > 2)."""
