import aisynphys
from aisynphys.database import SynphysDatabase

# SET CACHE FILE LOCATION FOR DATASET DOWNLOAD:
aisynphys.config.cache_path = "/tungstenfs/scratch/gzenke/rossjuli/datasets"

# WARNING: DOWNLOADS THE FULL 180 GB DATASET
db = SynphysDatabase.load_version('synphys_r1.0_2019-08-29_full.sqlite')
