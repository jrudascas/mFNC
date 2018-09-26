import matplotlib

#matplotlib.use('Agg')

import matplotlib.pyplot as plt

#plt.ioff()

import utils as u
import core as c
from nilearn import plotting
import os
import numpy as np
import plotGallery as pg
import csv

TR = 2.46
lag = 3
new_tr = 0.5

namesNodes_node_to_node = ['Auditory', 'Cerebellum', 'DMN', 'ECL', 'ECR', 'Salience', 'SensoriMotor', 'Vis_Lateral',
                           'Vis_Medial', 'Vis_Occipital']

crs_r_hc = '/home/jrudascas/Desktop/Test/crs_r_hc.csv'
crs_r_mcs = '/home/jrudascas/Desktop/Test/crs_r_mcs.csv'
crs_r_uws = '/home/jrudascas/Desktop/Test/crs_r_uws.csv'

crs_r = '/home/jrudascas/Desktop/Test/crs_r.csv'

crs_r_hc_values = []
crs_r_mcs_values = []
crs_r_uws_values = []

with open(crs_r_hc) as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        crs_r_hc_values.append(int(row[1]))

with open(crs_r_mcs) as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        crs_r_mcs_values.append(int(row[1]))

with open(crs_r_uws) as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        crs_r_uws_values.append(int(row[1]))

#path_general = '/home/runlab/data/COMA/'
#path_general_atlas = '/home/runlab/data/Atlas/RSN/'
path_general = '/home/jrudascas/Desktop/Test/'
#path_general = '/home/jrudascas/Desktop/Oa/'

path_general_atlas = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/RSN/'

list_path_atlas = [path_general_atlas + 'frAuditory_corr.nii.gz',
                   path_general_atlas + 'frCerebellum_corr.nii.gz',
                   path_general_atlas + 'frDMN_corr.nii.gz',
                   path_general_atlas + 'frECN_L_corr.nii.gz',
                   path_general_atlas + 'frECN_R_corr.nii.gz',
                   path_general_atlas + 'frSalience_corr.nii.gz',
                   path_general_atlas + 'frSensorimotor_corr.nii.gz',
                   path_general_atlas + 'frVisual_lateral_corr.nii.gz',
                   path_general_atlas + 'frVisual_medial_corr.nii.gz',
                   path_general_atlas + 'frVisual_occipital_corr.nii.gz']

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
                print(1)
                #if os.path.join(path_subject, 'time_series.txt'):
                time_series_rsn = u.to_extract_time_series(path_full_file, list_path_altas=list_path_atlas)
                print(2)
                connectivity_matrix, td_matrix, awtd_matrix, tr = core.run2(time_series_rsn, tr=TR, lag=lag, new_tr=new_tr)
                print(3)
                list_connectivity_matrixs.append(connectivity_matrix)
                list_td_matrixs.append(td_matrix*tr)

        fig, ax = plt.subplots()
        plotting.plot_matrix(np.mean(np.array(list_connectivity_matrixs), axis=0), labels=namesNodes_node_to_node,
                             vmax=1, vmin=-1, figure=fig)
        fig.savefig(path_group + group + '_mean.png', dpi=600)

        list_connectivity_matrixs_group.append(list_connectivity_matrixs)
        list_td_matrixs_group.append(list_td_matrixs)

td_hc = np.array(list_td_matrixs_group[0])[:,np.triu_indices(10, k=1)[0], np.triu_indices(10, k=1)[1]]
td_mcs = np.array(list_td_matrixs_group[1])[:,np.triu_indices(10, k=1)[0], np.triu_indices(10, k=1)[1]]
td_uws = np.array(list_td_matrixs_group[2])[:,np.triu_indices(10, k=1)[0], np.triu_indices(10, k=1)[1]]

pg.fivethirtyeightPlot(td_mcs, td_uws, group3=td_hc, lag=lag, save='ThreadsLagPC.png')

print("Test Graph MCS UWS\n")
pList1 = u.toFindStatisticDifference(u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[1])),
                                     u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[2])),
                                     is_corrected=True)

print("Test Graph HC MCS\n")
pList1 = u.toFindStatisticDifference(u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[0])),
                                     u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[1])),
                                     is_corrected=True)

print("Test Graph HC UWS\n")
pList1 = u.toFindStatisticDifference(u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[0])),
                                     u.buildFeaturesVector(np.array(list_connectivity_matrixs_group[2])),
                                     is_corrected=True)

print("\nLaggeds HC MCS")
pList1 = u.toFindStatisticDifference(np.mean(np.array(list_td_matrixs_group[0]), axis=-1),
                                     np.mean(np.array(list_td_matrixs_group[1]), axis=-1))

print("\nLaggeds HC UWS")
pList1 = u.toFindStatisticDifference(np.mean(np.array(list_td_matrixs_group[0]), axis=-1),
                                     np.mean(np.array(list_td_matrixs_group[2]), axis=-1))

print("\nLaggeds MCS UWS")
pList1 = u.toFindStatisticDifference(np.mean(np.array(list_td_matrixs_group[1]), axis=-1),
                                     np.mean(np.array(list_td_matrixs_group[2]), axis=-1))


print(np.mean(np.array(list_td_matrixs_group[2]), axis=-1).shape)

for index in range(np.mean(np.array(list_td_matrixs_group[2]), axis=-1).shape[-1]):
    u.plot_linear_regression((crs_r_hc_values, crs_r_mcs_values, crs_r_uws_values),
                         (np.mean(np.array(list_td_matrixs_group[0]), axis=-1)[:,index],
                          np.mean(np.array(list_td_matrixs_group[1]), axis=-1)[:,index],
                          np.mean(np.array(list_td_matrixs_group[2]), axis=-1)[:,index]), str(index))

np.savetxt(path_general + 'mean_td_hc.txt', np.mean(np.array(list_td_matrixs_group[0]), axis=-1), delimiter=' ',
           fmt='%s')
np.savetxt(path_general + 'mean_td_mcs.txt', np.mean(np.array(list_td_matrixs_group[1]), axis=-1), delimiter=' ',
           fmt='%s')
np.savetxt(path_general + 'mean_td_uws.txt', np.mean(np.array(list_td_matrixs_group[2]), axis=-1), delimiter=' ',
           fmt='%s')
