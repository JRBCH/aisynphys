import aisynphys
from aisynphys.database import SynphysDatabase
from aisynphys.cell_class import CellClass, classify_cells, classify_pairs
from aisynphys.dynamics import *

import pandas
import matplotlib.pyplot as plt
import seaborn as sns

# SET CACHE FILE LOCATION FOR DATASET DOWNLOAD:
aisynphys.config.cache_path = "/tungstenfs/scratch/gzenke/rossjuli/datasets"

# WARNING: DOWNLOADS THE FULL 180 GB DATASET
db = SynphysDatabase.load_version('synphys_r1.0_2019-08-29_full.sqlite')

# Load all synapses associated with mouse V1 projects
pairs = db.pair_query(synapse=True).all()

print("loaded %d synapses" % len(pairs))

cell_classes = {
    'pyr': CellClass(cell_class='ex', name='pyr'),
    'pvalb': CellClass(cre_type='pvalb', name='pvalb'),
    'sst': CellClass(cre_type='sst', name='sst'),
    'vip': CellClass(cre_type='vip', name='vip'),
}

# get a list of all cells in the selected pairs
cells = set([pair.pre_cell for pair in pairs] + [pair.post_cell for pair in pairs])

# Classify each cell. Note that, depending on the class definitions above, a cell could
# belong to multiple classes.
cell_class = {}
for cell in cells:
    # which of the classes defined above is this cell a member of?
    cell_in_classes = [cls_name for cls_name, cls in cell_classes.items() if cell in cls]
    cell_class[cell] = ','.join(cell_in_classes)

# construct a pandas dataframe containing the pre/postsynaptic cell class names
# and a measure of short-term plasticity
pre_class = [cell_class[pair.pre_cell] for pair in pairs]
post_class = [cell_class[pair.post_cell] for pair in pairs]
stp = [None if pair.dynamics is None else pair.dynamics.stp_induction_50hz for pair in pairs]

df = pandas.DataFrame(
    zip(pairs, pre_class, post_class, stp),
    columns=['pair', 'pre_class', 'post_class', 'stp'])

# select out only cells that are a member of exactly 1 class
mask = df.pre_class.isin(cell_classes) & df.post_class.isin(cell_classes)
df = df[mask]

# select only pairs with a measured stp
df = df.dropna()

stp = df.pivot_table('stp', 'pre_class', 'post_class', aggfunc=np.mean)
count = df.pivot_table('stp', 'pre_class', 'post_class', aggfunc=len)

# sort rows/cols into the expected order
order = list(cell_classes)
stp = stp[order].loc[order]

fig, ax = plt.subplots(figsize=(8, 6))

hm = sns.heatmap(stp, cmap='coolwarm', vmin=-0.4, vmax=0.4, square=True, ax=ax,
                 cbar_kws={"ticks": [-0.3, 0, 0.3], 'label': '<-- depressing       facilitating -->'})

fig.suptitle("50 Hz Train-induced STP", fontsize=16)
hm.set_xlabel("postsynaptic", fontsize=14)
hm.set_ylabel("presynaptic", fontsize=14)
hm.figure.axes[-1].yaxis.label.set_size(14)
hm.tick_params(labelsize=12)

plt.show()