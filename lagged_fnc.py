import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

import utils as u
import core as c
from nilearn import plotting
import os
import numpy as np

TR = 2.46
lag = 4

namesNodes_node_to_node = ['Auditory', 'Cerebellum', 'DMN', 'ECL', 'ECR', 'Salience', 'SensoriMotor', 'Vis_Lateral',
                           'Vis_Medial', 'Vis_Occipital']

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

path_general = '/home/runlab/data/COMA/'
name_file = 'data/functional/fmriGLM.nii.gz'

for group in sorted(os.listdir(path_general)):
    path_group = os.path.join(path_general, group)
    if os.path.isdir(path_group):
        list_connectivity_matrixs = []
        for dir in sorted(os.listdir(path_group)):
            path_subject = os.path.join(os.path.join(path_general, group), dir)
            if os.path.isdir(path_subject):
                print(dir)
                path_full_file = os.path.join(path_subject, name_file)

                time_series_rsn = u.to_extract_time_series(path_full_file, list_path_altas=list_path_atlas)

                connectivity_matrix, td_matrix, awtd_matrix = core.run2(time_series_rsn, tr=TR, lag=lag)

                list_connectivity_matrixs.append(connectivity_matrix)

        fig, ax = plt.subplots()
        fig = plotting.plot_matrix(np.mean(np.array(list_connectivity_matrixs), axis=0), labels=namesNodes_node_to_node, vmax=1, vmin=-1, figure=fig)
        fig.savefig(path_group + group + '_mean.png', dpi=600)


