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