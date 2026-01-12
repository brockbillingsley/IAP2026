# Investigation of the data
We are considering a toy model of $B^0\to K\pi\mu\mu$ located here:
```
/ceph/submit/data/user/a/anbeck/B2KPiMM_michele/full.root
```
(Note that I changed the path wrt. earlier to avoid having this huge long name.)
It is convenient to define filepaths like
```
export DATADIR=/ceph/submit/data/user/a/anbeck/B2KPiMM_michele
```
to avoid typing out the full path everytime which makes the commands huge. The location can be accessed using $DATADIR. The `export` command has to be run every time a new terminal is opened.

The `plotter.py` script looks at the different distributions and data inside this file. It can be run using
```bash
export DATADIR=/ceph/submit/data/user/a/anbeck/B2KPiMM_michele
python plotter.py --data $DATADIR/full.root
```
