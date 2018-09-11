import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

import utils as u
import core as c
from nilearn import plotting
import os
import numpy as np
import plotGallery as pg

TR = 2.46
lag = 3

namesNodes_node_to_node = ['Auditory', 'Cerebellum', 'DMN', 'ECL', 'ECR', 'Salience', 'SensoriMotor', 'Vis_Lateral',
                           'Vis_Medial', 'Vis_Occipital']

path_general = '/home/runlab/data/COMA/'

list_path_atlas = ['/home/runlab/data/Atlas/RSN/frAuditory_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frCerebellum_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frDMN_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frECN_L_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frECN_R_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frSalience_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frSensorimotor_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frVisual_lateral_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frVisual_medial_corr.nii.gz',
                   '/home/runlab/data/Atlas/RSN/frVisual_occipital_corr.nii.gz']

core = c.Core()

name_file = 'data/functional/fmriGLM.nii.gz'

list_connectivity_matrixs_group = []
list_td_matrixs_group = []

for group in sorted(os.listdir(path_general)):
    path_group = os.path.join(path_general, group)
    if os.path.isdir(path_group):
        list_connectivity_matrixs = []
        list_td_matrixs = []
        for dir in sorted(os.listdir(path_group)):
            path_subject = os.path.join(os.path.join(path_general, group), dir)
            if os.path.isdir(path_subject):
                print(dir)
                path_full_file = os.path.join(path_subject, name_file)

                #time_series_rsn = u.to_extract_time_series(path_full_file, path_general, group, dir, list_path_altas=list_path_atlas)
                time_series_rsn = u.to_extract_time_series(path_full_file, list_path_altas=list_path_atlas)

                connectivity_matrix, td_matrix, awtd_matrix, tr = core.run2(time_series_rsn, tr=TR, lag=lag)

                list_connectivity_matrixs.append(connectivity_matrix)
                list_td_matrixs.append(td_matrix)

        fig, ax = plt.subplots()
        plotting.plot_matrix(np.mean(np.array(list_connectivity_matrixs), axis=0), labels=namesNodes_node_to_node, vmax=1, vmin=-1, figure=fig)
        fig.savefig(path_group + group + '_mean.png', dpi=600)

        list_connectivity_matrixs_group.append(list_connectivity_matrixs)
        list_td_matrixs_group.append(list_td_matrixs)

print(np.array(list_td_matrixs_group[0]).shape)
print(np.triu_indices(10, k=1))

#td_hc = np.array(list_td_matrixs_group[0])[:,np.triu_indices(10, k=1)[0], np.triu_indices(10, k=1)[1]]
#td_mcs = np.array(list_td_matrixs_group[1])[:,np.triu_indices(10, k=1)[0], np.triu_indices(10, k=1)[1]]
#td_uws = np.array(list_td_matrixs_group[2])[:,np.triu_indices(10, k=1)[0], np.triu_indices(10, k=1)[1]]

#pg.fivethirtyeightPlot(td_mcs, td_uws, group3=td_hc, lag=lag, save='ThreadsLagPC.png')

print("Test Graph MCS UWS\n")
pList1 = u.toFindStatisticDifference(u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[1])), u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[2])),
                                         measure='manwhitneyu', is_corrected=True)

print("Test Graph HC MCS\n")
pList1 = u.toFindStatisticDifference(u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[0])), u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[1])),
                                         measure='manwhitneyu', is_corrected=True)

print("Test Graph HC UWS\n")
pList1 = u.toFindStatisticDifference(u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[0])), u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[2])),
                                         measure='manwhitneyu', is_corrected=True)

print("\nLaggeds HC MCS")
print(np.array(list_td_matrixs_group[0]).shape)

pList1 = u.toFindStatisticDifference(np.mean(np.array(np.array(list_td_matrixs_group[0])), axis=-1), np.mean(np.array(list_td_matrixs_group[1]), axis=-1),
                                         measure='manwhitneyu', is_corrected=True)

print("\nLaggeds HC UWS")
pList1 = u.toFindStatisticDifference(np.mean(np.array(np.array(list_td_matrixs_group[0])), axis=-1), np.mean(np.array(list_td_matrixs_group[2]), axis=-1),
                                         measure='manwhitneyu', is_corrected=True)

print("\nLaggeds MCS UWS")
pList1 = u.toFindStatisticDifference(np.mean(np.array(np.array(list_td_matrixs_group[1])), axis=-1), np.mean(np.array(list_td_matrixs_group[2]), axis=-1),
                                         measure='manwhitneyu', is_corrected=True)