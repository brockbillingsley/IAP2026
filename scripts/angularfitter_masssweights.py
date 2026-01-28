import numpy as np  # Numerical library
import yaml  # For reading YAML files
import uproot  # For reading ROOT files
import matplotlib.pyplot as plt  # Plotting library
import zfit  # Fitting library
import hist  # Histogram library
import json  # For reading JSON files
from myconstants import *
import tools  # Some helpful functions
import mypdfs  # Custom pdfs
import angularfunctions as af  # Angular functions
import h5py
import os
# from hepstats.splot import compute_sweights  # For sWeights computation

# Makes nice default plots
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

np.random.seed(0)
zfit.settings.set_seed(0)
zfit.settings.set_verbosity(10)

import pandas as pd

def eval_pdf(model, x, params=None, allow_extended=False):
    """Compute pdf of model at a given point x and for given parameters values"""

    if params is None:
        params = {}

    def pdf(model, x):
        ret = model.ext_pdf(x) if model.is_extended and allow_extended else model.pdf(x)

        return np.array(ret)

    for param in model.get_params():
        if param in params:
            value = params[param]["value"]
            param.set_value(value)
    return pdf(model, x)


def compute_sweights(model, x, weights=None):
    if weights is None:
        weights = np.ones(len(x))

    models = model.get_models()
    yields = [m.get_yield() for m in models]

    # p = np.vstack([np.array(m) for m in models]).T
    # Nx = np.array(model.ext_pdf(data))*weights
    p = np.vstack([eval_pdf(m, x)*weights for m in models]).T
    Nx = eval_pdf(model, x, allow_extended=True)*weights
    pN = p / Nx[:, None]

    Vinv = (pN).T.dot(pN)
    V = np.linalg.inv(Vinv)

    sweights = p.dot(V) / Nx[:, None]

    return {y: sweights[:, i] for i, y in enumerate(yields)}

def _pick_col(df, candidates, label):
    """Return first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {label} in H5. Tried: {candidates}. Available: {list(df.columns)}")

def _weighted_hist(values, weights, edges):
    """Histogram returning counts per bin (not /binwidth)."""
    h, _ = np.histogram(values, bins=edges, weights=weights)
    return h.astype(float)

def _scale_to_match(sum_target, sum_source):
    if sum_source == 0:
        return 0.0
    return float(sum_target) / float(sum_source)

def _scale_ref_to_match_data(ref_raw, data_counts, bin_width):
    """
    ref_raw: reference array loaded from H5 (either counts/bin OR density)
    data_counts: histogram bin contents in *counts* (sum of weights per bin)
    bin_width: float

    Returns: ref_y in the SAME units as plot (i.e. density = counts/bin_width)
    """
    # What points represent in total counts (sum of weights):
    target_total = float(np.sum(data_counts))

    # Two hypotheses for what ref_raw means:
    # H1: ref_raw is COUNTS per bin
    ref_total_counts = float(np.sum(ref_raw))

    # H2: ref_raw is already a DENSITY (counts/bin_width), so total counts ~ sum(density)*bin_width
    ref_total_from_density = float(np.sum(ref_raw) * bin_width)

    # Pick the interpretation that better matches the target total
    err_counts = abs(ref_total_counts - target_total)
    err_density = abs(ref_total_from_density - target_total)

    if err_density < err_counts:
        # ref_raw is density; scale density so that integral matches target_total
        scale = target_total / ref_total_from_density if ref_total_from_density > 0 else 1.0
        ref_y = ref_raw * scale              # still density
    else:
        # ref_raw is counts; scale counts to target, then convert to density
        scale = target_total / ref_total_counts if ref_total_counts > 0 else 1.0
        ref_counts = ref_raw * scale
        ref_y = ref_counts / bin_width       # convert to density to match plot

    return ref_y

def _read_ref_components_h5(ref_h5):
    """
    Read reference component weights + variables from an HDF5 file.

    Works for pandas "fixed" HDF layout using only h5py (no pytables needed).

    Returns:
        wS_ref, wA0_ref, wA1_ref, ref_vals_dict
    """
    import numpy as np
    import h5py

    def _dec(x):
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

    # --- helper: pick a key from dict by candidate names (case-insensitive + substring) ---
    def _pick_key(d, candidates):
        keys = list(d.keys())
        low = {k.lower(): k for k in keys}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        for c in candidates:
            for k in keys:
                if c.lower() in k.lower():
                    return k
        raise KeyError(f"Could not find any of {candidates} among keys: {keys}")

    with h5py.File(ref_h5, "r") as f:
        # Expect pandas fixed DataFrame under group "data"
        if "data" not in f:
            raise KeyError(f"Expected group 'data' in {ref_h5}. Top-level keys: {list(f.keys())}")

        g = f["data"]

        # columns live in axis0
        axis0 = [_dec(x) for x in g["axis0"][...]]  # all columns (may include all blocks)
        # axis1 is index (not needed unless debugging)
        # axis1 = g["axis1"][...]

        # Reconstruct columns from each block
        ref_vals = {}
        i = 0
        while f"block{i}_items" in g:
            items = [_dec(x) for x in g[f"block{i}_items"][...]]
            vals = np.asarray(g[f"block{i}_values"][...])

            # pandas stores values as (ncols_in_block, nrows) -> transpose to (nrows, ncols)
            if vals.ndim == 2 and vals.shape[0] == len(items):
                vals = vals.T

            for j, name in enumerate(items):
                ref_vals[name] = vals[:, j].astype(float) if np.issubdtype(vals[:, j].dtype, np.number) else vals[:, j]
            i += 1

    # ---- Now pick the weight columns from ref_vals ----
    # You may need to tweak candidates after you print columns (see command above).
    wS_key  = _pick_key(ref_vals, ["wS", "wAS", "w_refS", "wSig", "w_sig", "wsig"])
    wA0_key = _pick_key(ref_vals, ["wA0", "w0", "w_refA0"])
    # In this ref file, "A1" is stored as "App"
    wA1_key = _pick_key(ref_vals, ["wA1", "wApp", "w1", "w_refA1", "w_refApp"])


    wS_ref  = np.asarray(ref_vals[wS_key],  dtype=float)
    wA0_ref = np.asarray(ref_vals[wA0_key], dtype=float)
    wA1_ref = np.asarray(ref_vals[wA1_key], dtype=float)

    # Normalize naming: downstream code expects "wA1"
    ref_vals["wA1"] = wA1_ref

    return wS_ref, wA0_ref, wA1_ref, ref_vals


def _components_to_fracs(wS, wA0, wA1):
    """
    Convert component weights -> per-event fractions that sum to 1.
    Clip negatives to 0 (safety).
    """
    wS  = np.clip(wS,  0.0, np.inf)
    wA0 = np.clip(wA0, 0.0, np.inf)
    wA1 = np.clip(wA1, 0.0, np.inf)

    denom = wS + wA0 + wA1
    denom_safe = np.where(denom > 0, denom, 1.0)

    fS  = wS  / denom_safe
    fA0 = wA0 / denom_safe
    fA1 = wA1 / denom_safe

    mask0 = denom <= 0
    fS[mask0] = 0.0
    fA0[mask0] = 0.0
    fA1[mask0] = 0.0

    return fS, fA0, fA1

args = tools.parser()
if getattr(args, "settings", None) is None:
    raise RuntimeError("Missing --settings. Example: --settings plots/.../results/0.yml or a .yml/.json settings file.")
if getattr(args, "data", None) is None:
    raise RuntimeError("Missing --data. Example: --data outputs/combined_with_sweights.root")



if args.toy:
    name = "toy"
else:
    name = "data"


if len(args.fix_to_zero) > 0:
    for n in args.fix_to_zero:
        name += f"_{n}=0"
if len(args.fix_to_value) > 0:
    for n in range(0, len(args.fix_to_value), 2):
        name += f"_{args.fix_to_value[n]}={args.fix_to_value[n+1]}"
if len(args.fix_to_truth) > 0:
    for n in args.fix_to_truth:
        name += f"_{n}"
if len(args.constrain) > 0:
    for n in constrain_list:
        name += f"_{n}=constraint"


if len(args.qsq) == 2:
    name += f"_qsq-{args.qsq[0]}-{args.qsq[1]}"

tools.makedirs(args.polynomial, name)

# limits for integrals
limith = zfit.Space(axes=0, lower=-1, upper=1)
limitl = zfit.Space(axes=1, lower=-1, upper=1)
limits = limith * limitl

# create phsp
cosh = zfit.Space('cosh', limits=(-1, 1))
cosl = zfit.Space('cosl', limits=(-1, 1))
angles = cosh * cosl

# true values
# check if json or yaml
if args.settings.endswith(".yml"):
    with open(args.settings) as f:
        truth = yaml.load(f, Loader=yaml.FullLoader)
else:
    with open(args.settings) as f:
        truth = json.load(f)
        for t in truth:
            truth[t] = {"value": truth[t]}

for zi in args.fix_to_zero:
    truth[zi]["value"] = 0
for i in range(0, len(args.fix_to_value), 2):
    truth[args.fix_to_value[i]]["value"] = float(args.fix_to_value[i+1])

if args.toy:
    ntoys = 100
    nbins = 30
else:
    ntoys = 1
    nbins = 50

# Initialize parameters
App = zfit.Parameter("App", 0.1670, -1.0, 2.0)
A0 = zfit.Parameter("A0", 0.5, -1.0, 2.0)
Aqs = zfit.Parameter("Aqs", 0.01, -10.0, 10.0)
Aqc = zfit.Parameter("Aqc", 0.01, -10.0, 10.0)
AfbHS = zfit.Parameter("AfbHS", 0.0, -1.0, 1.0)
AfbHC = zfit.Parameter("AfbHC", 0.0, -1.0, 1.0)
AfbLS = zfit.Parameter("AfbLS", 0.0, -1.0, 1.0)
AfbLC = zfit.Parameter("AfbLC", 0.0, -1.0, 1.0)

# Set to the true values if provided
if "App" in truth.keys():
    App.set_value(truth["App"]["value"])
if "A0" in truth.keys():
    A0.set_value(truth["A0"]["value"])
if "Aqs" in truth.keys():
    Aqs.set_value(truth["Aqs"]["value"])
if "Aqc" in truth.keys():
    Aqc.set_value(truth["Aqc"]["value"])
if "AfbHS" in truth.keys():
    AfbHS.set_value(truth["AfbHS"]["value"])
if "AfbHC" in truth.keys():
    AfbHC.set_value(truth["AfbHC"]["value"])
if "AfbLS" in truth.keys():
    AfbLS.set_value(truth["AfbLS"]["value"])
if "AfbLC" in truth.keys():
    AfbLC.set_value(truth["AfbLC"]["value"])


def ASconditions(params):
    # The sum of all amplitudes must be 1.
    # This means that AS is not a free parameter.
    return 1-params['A0']-params['App']-params['Aqc']-params['Aqs']


AS = zfit.ComposedParameter("AS", ASconditions,
                            params={'A0': A0, 'App': App, 'Aqc': Aqc, 'Aqs': Aqs})

# total yield
Nsig = zfit.Parameter("Nsig", truth['Nsig']["value"], 0.0, 1.0e8)


# component yields
def yieldAS(params):
    # S-wave yield
    return params['Nsig']*params['AS']


def yieldApp(params):
    # Perp/parallel yield
    return params['Nsig']*params['App']


def yieldA0(params):
    # 0 yield
    return params['Nsig']*params['A0']


def yieldAq(params):
    # beta-dependent yield
    return params['Nsig']*(params['Aqc']+params['Aqs'])


def yieldP(params):
    # P-wave yield
    return params['Nsig'] - params['N_AS']


# Define the yields as composed parameters based on the total yield
N_AS = zfit.ComposedParameter("N_AS", yieldAS,
                              params={'Nsig': Nsig, 'AS': AS})
N_App = zfit.ComposedParameter("N_App", yieldApp,
                               params={'Nsig': Nsig, 'App': App})
N_A0 = zfit.ComposedParameter("N_A0", yieldA0,
                              params={'Nsig': Nsig, 'A0': A0})
N_Aq = zfit.ComposedParameter("N_Aq", yieldAq,
                              params={'Nsig': Nsig, 'Aqc': Aqc, 'Aqs': Aqs})
N_P = zfit.ComposedParameter("N_P", yieldP,
                             params={'Nsig': Nsig, 'N_AS': N_AS})


# Create the pdf and register the analytic integral (this makes the computations faster)
fitpdf = mypdfs.my2Dpdf(obs=angles, App=App, A0=A0, AS=AS, Aqc=Aqc, Aqs=Aqs,
                        AfbHC=AfbHC, AfbHS=AfbHS, AfbLC=AfbLC, AfbLS=AfbLS)
fitpdf.register_analytic_integral(func=mypdfs.integral, limits=limits)

# pdfs are normalized to an integral of 1 but we want extended pdfs


# Apply constraints or fix parameters if requested
constraints = []

# Loop through all parameters
for p in fitpdf.get_params():
    if p.name in args.fix_to_zero:
        # Set parameter to zero
        p.floating = False
        p.set_value(0)
    if p.name in args.fix_to_value:
        # Set parameter to a specific value
        p.floating = False
        p.set_value(float(args.fix_to_value[args.fix_to_value.index(p.name)+1]))
    if p.name in args.fix_to_truth:
        # Fix parameter to its true value
        p.floating = False
        p.set_value(truth[p.name]["value"])
    if p.name in args.constrain:
        # Constrain parameter to its true value with a Gaussian constraint
        observed = truth[p.name]["value"]
        sigma = max(abs(truth[p.name]["error_lower"]), abs(truth[p.name]["error_upper"]))
        constraints.append(zfit.constraint.GaussianConstraint(p, observation=observed, sigma=sigma))


# create pdfs for sWeights (no asymmetry terms!)
pdfS = mypdfs.my2Dpdf_AS(obs=angles)
pdfS = pdfS.create_extended(N_AS)
pdfApp = mypdfs.my2Dpdf_App(obs=angles)
pdfApp.register_analytic_integral(func=mypdfs.integral_App, limits=limits)
pdfApp = pdfApp.create_extended(N_App)
pdfA0 = mypdfs.my2Dpdf_A0(obs=angles)
pdfA0.register_analytic_integral(func=mypdfs.integral_A0, limits=limits)
pdfA0 = pdfA0.create_extended(N_A0)
pdfAq = mypdfs.my2Dpdf_Aq(obs=angles, Aqc=Aqc, Aqs=Aqs)
pdfAq.register_analytic_integral(func=mypdfs.integral_Aq, limits=limits)
pdfAq = pdfAq.create_extended(N_Aq)

pdfsweightslist = []
if not (AS.name in args.fix_to_zero):
    pdfsweightslist.append(pdfS)
if not (App.name in args.fix_to_zero):
    pdfsweightslist.append(pdfApp)
if not (A0.name in args.fix_to_zero):
    pdfsweightslist.append(pdfA0)
if not (Aqs.name in args.fix_to_zero and Aqc.name in args.fix_to_zero):
    pdfsweightslist.append(pdfAq)
pdfsweights = zfit.pdf.SumPDF(pdfsweightslist)
pdfs = {m.get_yield(): m for m in pdfsweights.get_models()}


# Read the data
with uproot.open(args.data) as f:
    tree = f[args.tree]

    # Read what we need
    cols = ["cosThetaL", "cosThetaK", "mKpi", "q2", args.weight_branch]
    datai = tree.arrays(cols, library="pd")

    # Build the angular columns expected by the rest of the code
    datai["cosl"] = datai["cosThetaL"]
    datai["cosh"] = datai["cosThetaK"]

    # Apply the same cuts as before
    datai = datai.query(f"({args.mKpi[0]} < mKpi) & (mKpi < {args.mKpi[1]})")
    datai.dropna(inplace=True)
    datai = datai.query(f"({args.qsq[0]} < q2) & (q2 < {args.qsq[1]})")

    # Pull out weights
    weights = datai[args.weight_branch].to_numpy(dtype=float)

    print("Number of rows after cuts:", len(datai))
    print("Weight branch: ", args.weight_branch)
    print("weights min/max: ", float(weights.min()), float(weights.max()))
    print("sum(weights): ", float(weights.sum()))


# Select requested number of data points and ranges
if args.toy:
    if len(args.binned) == 2:
        # Toy in bins
        frac = len(datai.query(f"({args.binned[0]}<q2) & (q2<{args.binned[1]})"))/len(datai)
        print("Fraction of data in bin:", frac)
        truth["Nsig"]["value"] = int(args.nsig*frac)
    else:
        # Toy
        truth["Nsig"]["value"] = args.nsig
    Nsig.set_value(truth["Nsig"]["value"])
else:
    if len(args.binned) == 2:
        # Data in bins
        datai.query(f"({args.binned[0]}<q2) &(q2<{args.binned[1]})", inplace=True)
    weights = datai[args.weight_branch].to_numpy(dtype=float)
    data = zfit.Data.from_numpy(
        obs=angles,
        array=datai[["cosh", "cosl"]].to_numpy(dtype=float),
        weights=weights,
    )


print("Using weight branch:", args.weight_branch)
print("Available columns:", list(datai.columns))

print("weights summary:", float(weights.min()), float(weights.mean()), float(weights.max()))

# Load reference H5 (for overlay lines)
ref_df = pd.read_hdf(args.ref_h5, key="data")

ref_mKpi_col = _pick_col(ref_df, ["mKpi", "m_kpi", "mkpi"], "mKpi")
ref_q2_col   = _pick_col(ref_df, ["q2", "Q2"], "q2")

# Reference truth weights
ref_wS_col   = _pick_col(ref_df, ["wS", "w_sig", "wSig", "w_signal"], "wS (signal)")
ref_wA0_col  = _pick_col(ref_df, ["wA0", "w_a0", "wA_0"], "wA0")
# in the ref file it’s wApp (A_perp). treat as "A1" like legend
ref_wA1_col  = _pick_col(ref_df, ["wApp", "wA1", "wAperp", "wA_perp"], "wApp (A1/App)")

# Numpy views
ref_vals = {
    "mKpi": ref_df[ref_mKpi_col].to_numpy(dtype=float),
    "q2":   ref_df[ref_q2_col].to_numpy(dtype=float),
    "wS":   ref_df[ref_wS_col].to_numpy(dtype=float),
    "wA0":  ref_df[ref_wA0_col].to_numpy(dtype=float),
    "wA1":  ref_df[ref_wA1_col].to_numpy(dtype=float),
}

print("Loaded ref h5:", args.ref_h5, "cols:", list(ref_df.columns))

# Prepare for toys
pulls = {}
for p in fitpdf.get_params():
    if p.floating:
        pulls[p.name] = np.zeros(ntoys)
X = np.linspace(-1, 1, 100)


# Check that the pdf is well defined
assert (np.sum(fitpdf.pdf(data).numpy() <= 0) == 0)
assert (np.sum(np.isnan(fitpdf.pdf(data).numpy())) == 0)
assert (np.sum(np.isinf(fitpdf.pdf(data).numpy())) == 0)
assert (np.sum(np.isnan(np.log(fitpdf.pdf(data).numpy()))) == 0)
assert (np.sum(np.isinf(np.log(fitpdf.pdf(data).numpy()))) == 0)

# Save initial parameter values so we can safely reset after a failed fit
init_vals = {p.name: float(p.value()) for p in fitpdf.get_params()}

i = 0
while i < ntoys:
    print("Toy", i)
    seed = np.random.randint(0, 2**32-1)
    zfit.settings.set_seed(seed)
    np.random.seed(seed)

    # create minimizer
    if args.toy:
        minimizer = zfit.minimize.Minuit(strategy=zfit.minimize.DefaultToyStrategy)
    else:  # easier for debugging data
        minimizer = zfit.minimize.Minuit()

    if args.toy:
        NN = np.random.poisson(args.nsig)
        datatoy = datai.sample(n=NN, replace=True)
        if len(args.binned) == 2:
            datatoy.query(f"({args.binned[0]}<q2) &(q2<{args.binned[1]})", inplace=True)
        data = zfit.Data.from_pandas(datatoy[["cosh","cosl"]], obs=angles)
    else:
        datatoy = datai

    # Create the loss
    loss = zfit.loss.UnbinnedNLL(model=fitpdf, data=data)


    # Add constraints if any
    if len(constraints) > 0:
        loss.add_constraints(constraints)

    # Run the fit
    result = minimizer.minimize(loss)
    result.update_params()
    # try:
    #     print(result)
    #     result.errors()  # Compute uncertainty
    # except Exception as e:
    #     print(e)
    #     print("Problem with errors.")
    #     for p in fitpdf.get_params():
    #         if p.floating:
    #             print(p)
    #             p.set_value(init_vals[p.name])
    #             # p.randomize()
    #             print(p)
    #     continue
    try:
        print(result)
        result.hesse()          # fast symmetric errors
    except Exception as e:
        print(e)
        print("Problem with hesse/errors.")
        for p in fitpdf.get_params():
            if p.floating:
                print(p)
                p.set_value(init_vals[p.name])
                print(p)
        continue

    print(result)
    result.update_params()

    # --- Build per-event component fractions from angular model ---
    xdata = data  # zfit.Data with cosh/cosl and event weights already attached

    # component "extended contributions": Nk * pdf_k(x)
    t_AS  = eval_pdf(pdfs[N_AS],  xdata) * float(N_AS.value())
    t_A0  = eval_pdf(pdfs[N_A0],  xdata) * float(N_A0.value())
    t_App = eval_pdf(pdfs[N_App], xdata) * float(N_App.value())
    t_Aq  = eval_pdf(pdfs[N_Aq],  xdata) * float(N_Aq.value())

    t_sum = t_AS + t_A0 + t_App + t_Aq
    t_sum_safe = np.where(t_sum > 0, t_sum, 1.0)

    f_AS  = t_AS  / t_sum_safe
    f_A0  = t_A0  / t_sum_safe
    f_App = t_App / t_sum_safe
    f_Aq  = t_Aq  / t_sum_safe

    # --- Partition the mass-fit signal sWeights (args.weight_branch) ---
    wSig_mass = weights  # from ROOT branch

    wSig_AS  = wSig_mass * f_AS
    wSig_A0  = wSig_mass * f_A0
    wSig_App = wSig_mass * f_App
    wSig_Aq  = wSig_mass * f_Aq

    # event-by-event sanity:
    print("[check] mean(|sum - wSig|) =",
        float(np.mean(np.abs((wSig_AS+wSig_A0+wSig_App+wSig_Aq) - wSig_mass))))

    # Check that the result is valid
    covmat = result.covariance()
    posdef = np.all(np.linalg.eigvals(covmat) > 0)
    if not result.valid:
        print("Fit not valid.")
        for p in fitpdf.get_params():
            if p.floating:
                p.set_value(init_vals[p.name])
        continue

    # Save the fit results
    paramdict = {}
    pi = 0
    for p in result.params:
        paramdict[p.name] = {}
        paramdict[p.name]["value"] = result.params[p]["value"]
        # Robustly store uncertainties:
        # - MINOS gives: result.params[p]["errors"]["upper"/"lower"]
        # - HESSE gives: result.params[p]["error"] (symmetric)
        pinfo = result.params[p]

        if "errors" in pinfo and isinstance(pinfo["errors"], dict):
            paramdict[p.name]["error_upper"] = float(pinfo["errors"].get("upper", np.nan))
            paramdict[p.name]["error_lower"] = float(pinfo["errors"].get("lower", np.nan))
        elif "error" in pinfo:
            # symmetric error from HESSE
            err = float(pinfo["error"])
            paramdict[p.name]["error_upper"] = err
            paramdict[p.name]["error_lower"] = err
        else:
            # no uncertainties available
            paramdict[p.name]["error_upper"] = float("nan")
            paramdict[p.name]["error_lower"] = float("nan")
        paramdict[p.name]["covariance"] = {}
        qi = 0
        for q in result.params:
            if q == p:
                qi += 1
                continue
            paramdict[p.name]["covariance"][q.name] = float(covmat[pi][qi])
            qi += 1
        pi += 1

    outname = f"plots/{args.polynomial}/{name}/results/{i}.yml"
    with open(outname, 'w') as yaml_file:
        yaml.dump(paramdict, yaml_file, default_flow_style=False)

# Compute sWeights
    try:
        sweights = compute_sweights(pdfsweights, data)
    except Exception as e:
        print(e)
        print("Problem with massfit sweights.")
        continue

    # Sanity check
    diff = Nsig.value()-N_A0.value()-N_App.value()-N_Aq.value()-N_AS.value()
    assert (np.isclose(diff, 0, atol=1e-2))

    sApp, sA0, sAS, sAq = sweights[N_App], sweights[N_A0], sweights[N_AS], sweights[N_Aq]

    # Print the integrals after computing sWeights
    print("[check] sums:",
        "sum(sAS)", float(np.sum(sAS)), "N_AS", float(N_AS.value()),
        "sum(sA0)", float(np.sum(sA0)), "N_A0", float(N_A0.value()),
        "sum(sApp)", float(np.sum(sApp)), "N_App", float(N_App.value()),
        "sum(sAq)", float(np.sum(sAq)), "N_Aq", float(N_Aq.value()))
    print("[check] total sum sweights", float(np.sum(sAS+sA0+sApp+sAq)), "Nsig", float(Nsig.value()))

    # Save the pulls for toys
    if args.toy:
        for k in pulls.keys():
            par = result.params[k]
            if k in args.constrain:
                extra = max(abs(truth[k]["error_upper"]), abs(truth[k]["error_lower"]))
            else:
                extra = 0
            if par['value'] < truth[k]["value"]:
                pull = (truth[k]["value"] - par["value"])/np.sqrt(par["errors"]["upper"]**2+extra**2)
            else:
                pull = (truth[k]["value"] - par["value"])/np.sqrt(par["errors"]["lower"]**2+extra**2)
            pulls[k][i] = pull

    # Plot the result
    if i < 3:
        # Make the same type of plot for costhetah and costhetal
        for v, n, l in zip([cosh, cosl], ["cosh", "cosl"], [r"$\cos(\theta_h)$", r"$\cos(\theta_\ell)$"]):
            dist = v.limits[1][0][0] - v.limits[0][0][0]
            x = np.linspace(v.limits[0][0][0], v.limits[1][0][0], 1000)

            # Step 1) Plot the data
            H = hist.Hist(hist.axis.Regular(nbins, v.limits[0][0][0], v.limits[1][0][0], underflow=False, overflow=False))
            H.fill(datai[n].to_numpy(), weight=weights)
            mplhep.histplot(H, color='black', histtype='errorbar', label='Toy data', xerr=True, yerr=True, marker='.')

            # Step 2) Plot the individual components
            projAS = pdfs[N_AS].create_projection_pdf(obs=v)
            yAS = projAS.pdf(x).numpy() * float(N_AS.value()) * dist / nbins
            plt.fill_between(x, y1=np.zeros(len(yAS)), y2=yAS, facecolor='gold', alpha=0.6, label=r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$", linewidth=0, edgecolor="w", zorder=0, hatch='xx')
            projA0 = pdfs[N_A0].create_projection_pdf(obs=v)
            yA0 = projA0.pdf(x).numpy() * float(N_A0.value()) * dist / nbins
            plt.fill_between(x, y1=yAS, y2=yAS+yA0, color='navy', alpha=0.6, label=r"$n^P_0=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$", linewidth=0, zorder=0, edgecolor="w", hatch="//")
            projApp = pdfs[N_App].create_projection_pdf(obs=v)
            yApp = projApp.pdf(x).numpy() * float(N_App.value()) * dist / nbins
            plt.fill_between(x, y1=yAS+yA0, y2=yAS+yA0+yApp, color='dodgerblue', alpha=0.6, label=r"$n^P_1=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$", linewidth=0, zorder=0, edgecolor="w", hatch="\\\\")
            projAq = pdfs[N_Aq].create_projection_pdf(obs=v)
            yAq = projAq.pdf(x).numpy() * float(N_Aq.value()) * dist / nbins
            plt.fill_between(x, y1=yAS+yA0+yApp, y2=yAS+yA0+yApp+yAq, color='firebrick', alpha=0.6, label=r"$n_{\beta}$", linewidth=0, zorder=0, edgecolor="w", hatch="..")

            # Step 3) Plot the total fit
            projTot = fitpdf.create_projection_pdf(obs=v)
            Z = projTot.pdf(x).numpy() * float(Nsig.value()) * dist / nbins
            plt.plot(x, Z, label="Fit", color="k")

            # Step 4) Plot the interference terms (= asymmetries)
            if n == "cosh":
                yInt = (af.proj_AfbHC(x, n)*AfbHC.value() + af.proj_AfbHS(x, n)*AfbHS.value())*Nsig.value()*dist/nbins
                plt.fill_between(x, y1=np.zeros(len(yInt)), y2=yInt, color='gray', alpha=0.4, label=r'Interference', linewidth=0, zorder=0)
                plt.legend(loc='upper right', handlelength=1, handleheight=1, fontsize=24)
            else:
                yInt = (af.proj_AfbLC(x, n)*AfbLC.value() + af.proj_AfbLS(x, n)*AfbLS.value())*Nsig.value()*dist/nbins
                plt.fill_between(x, y1=np.zeros(len(yInt)), y2=yInt, color='gray', alpha=0.4, label=r'Interference', linewidth=0, zorder=0)

            # Some plotting settings
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.xlim(v.limits[0][0], v.limits[1][0])
            plt.xlabel(l, ha="right", x=1)
            plt.ylabel(fr"Data points / {(dist/nbins):.2f}",ha="right",y=1)
            plt.savefig(f"plots/{args.polynomial}/{name}/{i}_{n}.png", dpi=200)
            plt.close()


        # Also make weighted plots
        for vkey, l, u in zip(["mKpi", "q2"], [r"$m(K\pi)$", r"$q^2$"], [r"GeV$/c^2$", r"GeV$^2/c^4$"]):
            mi, ma = datatoy[vkey].min(), datatoy[vkey].max()
            dist = ma - mi
            H = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
            wSig_plot = datatoy["wSig"].to_numpy(dtype=float) if "wSig" in datatoy.columns else None
            xplot = datatoy[vkey].to_numpy()
            if wSig_plot is None:
                H.fill(xplot)
            else:
                H.fill(xplot, weight=wSig_plot)
            mplhep.histplot(H, color='black', histtype='errorbar', label='Toy data', xerr=True, yerr=True, marker='.', zorder=20)

            # component histograms from sweights
            H_AS = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
            H_A0 = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
            H_App = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
            H_Aq = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())

            vals = datatoy[vkey].to_numpy(dtype=float)

            H_AS.fill(vals, weight=wSig_AS)
            H_A0.fill(vals, weight=wSig_A0)
            H_App.fill(vals, weight=wSig_App)
            H_Aq.fill(vals, weight=wSig_Aq)

            mplhep.histplot(H_AS,  histtype="step", linewidth=1.4, linestyle="--", label="Data sAS")
            mplhep.histplot(H_A0,  histtype="step", linewidth=1.4, linestyle="--", label="Data sA0")
            mplhep.histplot(H_App, histtype="step", linewidth=1.4, linestyle="--", label="Data sApp")
            mplhep.histplot(H_Aq,  histtype="step", linewidth=1.4, linestyle="--", label="Data sAq")

            data_total = wSig_AS + wSig_A0 + wSig_App + wSig_Aq
            H_tot = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
            H_tot.fill(vals, weight=data_total)
            mplhep.histplot(H_tot, histtype="step", linewidth=2.0, label="Data TOTAL (sAS+sA0+sApp+sAq)")

            # Weighted histogram using *signal* mass sWeights only
            Hw = hist.Hist(
                hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False),
                storage=hist.storage.Weight()
            )
            Hw.fill(datatoy[vkey], weight=weights)

            mplhep.histplot(Hw, histtype='errorbar', label='Signal sWeighted', xerr=True, yerr=True, marker='.')

            # ---------- Reference overlay (from args.ref_h5) ----------
            edges = H.axes[0].edges
            centers = 0.5 * (edges[:-1] + edges[1:])

            ref_x = ref_vals[vkey]  # vkey is "mKpi" or "q2"

            # Reference component histograms in COUNTS per bin (sum of weights in each bin)
            ref_S  = _weighted_hist(ref_x, ref_vals["wS"],  edges)
            ref_A0 = _weighted_hist(ref_x, ref_vals["wA0"], edges)
            ref_A1 = _weighted_hist(ref_x, ref_vals["wA1"], edges)

            # Scale reference so that TOTAL ref yield matches TOTAL data yield in Hw
            target_total = float(np.sum(Hw.values()))  # total sum of weights in data hist
            ref_total = float(np.sum(ref_S + ref_A0 + ref_A1))

            ref_scale = (target_total / ref_total) if ref_total > 0 else 1.0

            ref_S  = ref_S  * ref_scale
            ref_A0 = ref_A0 * ref_scale
            ref_A1 = ref_A1 * ref_scale

            ref_sum = ref_S + ref_A0 + ref_A1
            plt.step(centers, ref_sum, where="mid", linestyle="-.", linewidth=1.8, label="Ref TOTAL (scaled)")

            print("[info] ref_scale =", ref_scale,
                "target_total =", target_total,
                "ref_total(before) =", ref_total)

            # Draw as dashed step lines (counts/bin)
            plt.step(centers, ref_S,  where="mid", linestyle="--", linewidth=1.6, label="Ref S (scaled)")
            plt.step(centers, ref_A0, where="mid", linestyle="--", linewidth=1.6, label="Ref A0 (scaled)")
            plt.step(centers, ref_A1, where="mid", linestyle="--", linewidth=1.6, label="Ref A1/App (scaled)")
            # ---------------------------------------------------------


            plt.legend(handletextpad=0.1, fontsize=24)
            plt.axhline(0, color='black', linewidth=1)
            plt.xlim(mi, ma)
            plt.xlabel(l + f" [{u}]", ha="right", x=1)
            plt.ylabel(r"$\sum$ weights / bin", ha="right", y=1)
            plt.savefig(f"plots/{args.polynomial}/{name}/{i}_{vkey}_weighted.png", dpi=200)
            plt.close()


    # Save the sWeighted data
    datas = data.to_pandas()
    datas["wSig"] = datatoy[args.weight_branch].to_numpy(dtype=float)
    datas['wS'] = sAS
    datas['wApp'] = sApp
    datas['wA0'] = sA0
    datas['wAq'] = sAq

    datas['mKpi'] = datatoy['mKpi'].values
    datas['q2'] = datatoy['q2'].values
    datas['cosl'] = datatoy['cosl'].values
    datas['cosh'] = datatoy['cosh'].values
    datas.to_hdf(f"sweights/{args.polynomial}/{name}/{i}.h5", key='data', mode='w')
    i += 1


# Plot the pull distributions if this was a toy study
if args.toy:
    mu = zfit.Parameter("mu", 0, -500, 500)
    sig = zfit.Parameter("sig", 1, 0, 100)
    x = zfit.Space('x', (-500, 500))
    gauss = zfit.pdf.Gauss(obs=x, mu=mu, sigma=sig)
    X = np.linspace(-5, 5, num=100)

    minimizer = zfit.minimize.Minuit()

    for k in pulls.keys():
        print("Pulls", k)
        pullsk = pulls[k]
        try:
            res = minimizer.minimize(loss=zfit.loss.UnbinnedNLL(model=gauss, data=zfit.data.Data.from_numpy(obs=x, array=pullsk)))
            result = res.hesse()
            # res.errors()
            print(result)
        except Exception as e:
            print(e)
            print("Problem with pull fit.")
            continue
        # plot data
        f = plt.figure()
        plt.figure(figsize=(f.get_size_inches()[0]/2,f.get_size_inches()[0]/2))
        mplhep.histplot(zfit.data.Data.from_numpy(obs=x, array=pullsk).to_binned(5000), color='black', histtype='errorbar', xerr=True, yerr=True, density=True)
        plt.plot(X, gauss.pdf(X), label=rf'$\mu={mu.value():.2f}({result[mu]["error"]:.2f})$'+'\n'+rf'$\sigma={sig.value():.2f}({result[sig]["error"]:.2f})$', color='red')
        plt.legend()
        plt.yticks([])
        plt.ylabel("Arbitrary Units")
        plt.xlim(-5, 5)
        plt.xlabel(fr'Pull of {labels[k]}')
        plt.savefig(f'plots/angularfit_2d/{args.polynomial}/{name}/pull_{k}.pdf')
        plt.close()

        mu.set_value(0)
        sig.set_value(1)
