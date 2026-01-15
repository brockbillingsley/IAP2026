The fitting script can be run using
```
python angularfitter.py --data $DATADIR/full.root --settings settings/App=0.1670_qsq-1.1-7.0.yml --no-toy --qsq 1.1 7 &> fitter.log
```
This is going to take a few minutes.

# Tasks
1. While the script is running, go through the file and try to understand the major steps.
2. Look at the `angularfunctions.py` and `mypdfs.py` files and identify the angular terms you can find in the first equations in [this paper](https://arxiv.org/pdf/2503.22549).
3. What outputs does the script produce?
4. What is inside `sweights/standard/data_qsq-1.1-7.0/0.h5`? Hints: it is created in L. 467 of the fitting script and you can open it using:
```
data = pd.read_hdf(filename)
```
5. Compare the sweighted distributions (plotting mKpi and q2 as histograms but with the weights found in the sweights file from task 4) to the reference samples in the DATADIR (there are references for A0, App, and AS, there is no reference for the beta-dependent terms, called nbeta in the paper and Aq in the fitting script).

# Additional tasks for people working on the inclusion of background
1. Go to the `genbkg` directory, generate a background toy sample, and look at the resulting distributions. Identify each shape in the generator script.
2. Create a new file in this directory here (`fitter`) called for example `backgroundfitter.py`. In this file
    i.  Define a two-dimensional pdf that describes the two angles cosThetaK and cosThetaL (you can steal this from `generator.py`).
    ii. Read the background sample you have just created (you can steal how this works from `angularfitter.py`).
    iii.    Fit the pdf you just created to the background data (steal from `angularfitter.py`).
    iv. Store the fit result (steal from `angularfitter.py`) and check that you got the correct parameters (they should be compatible with the ones you used to generate the toy sample).
    v. Create two figures, one for cosThetaK and one for cosThetaL, showing the data and the fit result (steal from `angularfitter.py`).