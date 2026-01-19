import zfit
import argparse
import json
import uproot

# If GeneralizedCB is not in core zfit, it may be in zfit_physics
try:
    GeneralizedCB = zfit.pdf.GeneralizedCB
except AttributeError:
    from zfit_physics.pdf import GeneralizedCB


parser = argparse.ArgumentParser(description="Generate signal events.")
parser.add_argument("--n", type=int, default=200000, help="Number of signal events to generate")
parser.add_argument("--output", type=str, default="sig_toy.root", help="Output ROOT file name")
parser.add_argument("--params", type=str, default="signal_params.json", help="JSON file with signal parameters")
args = parser.parse_args()

# Load parameters from JSON file
with open(args.params, 'r') as f:
    params_dict = json.load(f)["parameters"]

# Define the ranges of the variables (match your background generator)
qsq  = zfit.Space('q2',         limits=(1.1, 7.0))
mkpi = zfit.Space('mKpi',       limits=(0.65, 1.5))
bmass = zfit.Space('B_mass',    limits=(5.170, 5.500))
cosh = zfit.Space('cosThetaK',  limits=(-1.0, 1.0))
cosl = zfit.Space('cosThetaL',  limits=(-1.0, 1.0))

space = qsq * mkpi * bmass * cosh * cosl

# q2 and mKpi: keep simple (Uniform) unless you want something else
pdf_qsq  = zfit.pdf.Uniform(qsq.limits[0],  qsq.limits[1],  obs=qsq)
pdf_mkpi = zfit.pdf.Uniform(mkpi.limits[0], mkpi.limits[1], obs=mkpi)

# Signal mass: GeneralizedCB parameters from JSON
mu     = zfit.Parameter("mu",     params_dict["mu"])
sigmal = zfit.Parameter("sigmal", params_dict["sigmal"])
alphal = zfit.Parameter("alphal", params_dict["alphal"])
nl     = zfit.Parameter("nl",     params_dict["nl"])

sigmar = zfit.Parameter("sigmar", params_dict["sigmar"])
alphar = zfit.Parameter("alphar", params_dict["alphar"])
nr     = zfit.Parameter("nr",     params_dict["nr"])

pdf_bmass = GeneralizedCB(
    obs=bmass,
    mu=mu,
    sigmal=sigmal, alphal=alphal, nl=nl,
    sigmar=sigmar, alphar=alphar, nr=nr,
)

# Angles: simplest truth signal is Uniform
# (This is enough to validate that sWeights recover the correct angular shapes.)
pdf_cosh = zfit.pdf.Uniform(cosh.limits[0], cosh.limits[1], obs=cosh)
pdf_cosl = zfit.pdf.Uniform(cosl.limits[0], cosl.limits[1], obs=cosl)

# Combine into 5D signal PDF
pdf_sig = zfit.pdf.ProductPDF([pdf_qsq, pdf_mkpi, pdf_bmass, pdf_cosh, pdf_cosl], obs=space)

# Generate signal events
n_events = args.n
data_sig = pdf_sig.sample(n_events)

# Save to ROOT file
output_file = args.output
with uproot.recreate(output_file) as root_file:
    root_file["signal"] = {
        "q2": data_sig["q2"].numpy(),
        "mKpi": data_sig["mKpi"].numpy(),
        "B_mass": data_sig["B_mass"].numpy(),
        "cosThetaK": data_sig["cosThetaK"].numpy(),
        "cosThetaL": data_sig["cosThetaL"].numpy(),
    }

print(f"Generated {n_events} signal events and saved to {output_file}.")
