import numpy as np
import scipy.stats
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
pairs = db.pair_query(
    synapse=True,
    species='mouse',
    synapse_type='ex',
    acsf='1.3mM Ca & 1mM Mg',  # '1.3mM Ca & 1mM Mg' or '2mM Ca & Mg'
    electrical=False  # Exclude gap junctions
).all()

results = pandas.DataFrame(columns=['pair_id',
                                    'pre_cell',
                                    'post_cell',
                                    'rec_id',
                                    'clamp_mode',
                                    'ind_freq',
                                    'rec_delay',
                                    'amps']
                           )

for ix in range(len(pairs)):
    print('processing pair {} of {}'.format(ix + 1, len(pairs)))
    pair = pairs[ix]
    pr_recs = pulse_response_query(pair).all()  # extract all response trains

    # cull out all PRs that didn't get a fit
    pr_recs = [pr_rec for pr_rec in pr_recs if pr_rec.PulseResponseFit.fit_amp is not None]
    # sort by clamp mode and frequency
    sorted = sorted_pulse_responses(pr_recs)

    if sorted != {}:  # if any recordings have a fit

        # Loop through all recordings
        for key, recs in sorted.items():
            clamp_mode, ind_freq, rec_delay = key

            # only consider voltage clamp recordings
            # if clamp_mode != 'vc':
            #    continue

            for recording, pulses in recs.items():
                if 1 not in pulses or 2 not in pulses:
                    continue
                amps = {k: r.PulseResponseFit.fit_amp for k, r in pulses.items()}

                results = results.append({'pair_id': pair.id,
                                          'pre_cell': pair.pre_cell.cell_class,
                                          'post_cell': pair.post_cell.cell_class,
                                          'rec_id': recording.id,
                                          'clamp_mode': clamp_mode,
                                          'ind_freq': ind_freq,
                                          'rec_delay': rec_delay,
                                          'amps': list(amps.values())},
                                         ignore_index=True)

def calc_norm_amplitudes(results):

    initialAmp = {}
    # iterate over pairs
    for id in list(results.pair_id.unique()):
        subdat = results[results.pair_id==id]
        vcdat = subdat[subdat.clamp_mode == 'vc']
        icdat = subdat[subdat.clamp_mode == 'ic']
        initialAmp[id] = {'vc': np.array([vcdat.amps.iloc[i][0] for i in range(len(vcdat))]).mean(),
                          'ic': np.array([icdat.amps.iloc[i][0] for i in range(len(icdat))]).mean()}

    normAmp = []

    for i in range(len(results)):
        s = results.iloc[i]
        normAmp.append(np.array(s.amps) / initialAmp[s.pair_id][s.clamp_mode])

    results['normAmps'] = normAmp

    return results

def get_amps_first8(results, normalized=False):
    """
    Extracts response to first 8 pulses (no recovery pulses)
    """

    x = np.zeros((len(results), 8))

    for ix in range(len(results)):
        if normalized:
            amps = results.iloc[ix].normAmps
        else:
            amps = results.iloc[ix].amps

            if len(amps) >= 8:
                x[ix, :] = amps[:8]
            else:
                x[ix, :] = np.pad(amps, (0, 8-len(amps)), constant_values=np.nan)

    return x


# Example
x = get_amps_first8(results[(results.ind_freq==100.0) & (results.clamp_mode == 'vc')])

plt.plot(-x.T*(10**12), lw=0.2, color='gray')
plt.plot(np.nanmean(-x*(10**12), 0), lw=2, color='green')
plt.ylim(0, 75)
plt.xlabel('# stim')
plt.ylabel('EPSC amplitude (pA)')
plt.title('E-E 100Hz')

plt.savefig('example_EE_100Hz')
plt.show()

x = get_amps_first8(results[(results.ind_freq==50.0) & (results.clamp_mode == 'vc')])

plt.plot(-x.T*(10**12), lw=0.2, color='gray')
plt.plot(np.nanmean(-x*(10**12), 0), lw=2, color='green')
plt.ylim(0, 75)
plt.xlabel('# stim')
plt.ylabel('EPSC amplitude (pA)')
plt.title('E-E 50Hz')

plt.savefig('example_EE_50Hz')
plt.show()

# Example noise correlation analysis
x = get_amps_first8(results[(results.ind_freq == 20.0) & (results.clamp_mode == 'vc')])

