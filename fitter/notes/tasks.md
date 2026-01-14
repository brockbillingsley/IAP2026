# IAP2026 Fitter — Tasks 1–5

## Task 1 — Major steps
- reads input data from the command line (parser and args lines)
- sorts for .yml (yaml) files and initializes parameters used in the fit
- creates output directories (plots, results, sweights)
- reads the input root data and keeps necessary kinematic and angular variables
- applies q2 cuts (1.1 < q2 < 7.0 based on file names)
- defines the mathematical model (PDFs) for the variables
- peforms the fit
- computes sweights and saves (producing 0.h5)
- validates with plots
- saves final numerical outputs (results yml)

## Task 2 — Angular terms:
- cosl (y): cos(theta_l) lepton angle used in all four angular functions
- cosh (x): cos(theta_h) hadron angle used in all four angular functions
    all four angular functions seen are functions of these two variables
- Amplitudes that construct n coefficients:
    App: for perp/parallel function
    A0: for 0 function
    Aq: for beta-dependent function
    AS: for S-wave function
    AfbLS, AfbLC, AfbHS, AfbHC: asymmetry terms


## Task 3 — Script outputs
- Considering produced file type and order, as well as calling on knowledge of the process in question,
we can conclude that the script produces (i) a run log (/fitter.log), (ii) a YAML results file with fit outputs (.../results/0.yml), (iii) PDF diagnostic/validation plots (angular and weighted kinematic distributions; 4 pdf files for cosh, cosl, mKpi, and q2, individually), and (iv) an HDF5 file storing event-level sWeights for the selected q² bin.

## Task 4 — Contents of sweights/.../0.h5
- HDF keys:
- Columns:
- Which columns are weights:
- Head/describe summary:

## Task 5 — Weighted vs reference comparisons (mKpi, q2)
- Method:
- Plots saved to:
- A0 comparison notes:
- App comparison notes:
- AS comparison notes:
- Notes on Aq/nbeta (no reference):
