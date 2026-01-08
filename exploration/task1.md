# Task 1 - Data Inspection (Variables and Units)

From the initial "Open the ROOT file" section, I was able to load the ROOT
tree using uproot and I then inspected the resulting pandas dataframe.

## How I inspected the data
- Printed column names
- Printed first 5 rows
- Used 'describe()' to see typical ranges so that I could infer units

## Then, the variables returned and the units by my inference
- 'cosThetaK' (dimensionless): cos(theta_K), helicity-angle cosine for the K.
Range ~[-1, 1].
- 'cosThetaL' (dimensionless): helicity-angle cosine for the lepton. Range
~[-1,1].
- 'phi' (radians): azimuthal angle between the respective decay planes
- 'q2' (GeV^2): squared invariant mass of the dimuon system
- 'B_mass' (MeV): reconstructed invariant mass of the B candidate
- 'mKpi' (Gev): invariant mass of the Kpi system 

## Notes
The ROOT file did not provide explicit unit data, so units are inferred from
standard HEP conventions and observed value regions returned from the
'describe()' command.
