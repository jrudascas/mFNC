import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
#plt.ioff()

import utils as u
import core as c
from nilearn import plotting
import os
import numpy as np

TR = 2.46

atlas = {'AAN':'/home/jrudascas/Desktop/DWITest/Additionals/Atlas/2mm/AAN_2mm.nii', 'Cortex':'/home/jrudascas/Desktop/DWITest/Additionals/Atlas/2mm/HarvardOxford-cort-maxprob-thr25-2mm.nii'}

core = c.Core()

plotting.show()

path_general = '/media/jrudascas/HDRUDAS/alejandra/'
name_file = 'data/functional/fmriGLM.nii.gz'

list_group_td_map = []


for group in sorted(os.listdir(path_general)):
    path_group = os.path.join(path_general, group)

    if os.path.isdir(path_group):
        connectivity_matrixs = []
        for dir in sorted(os.listdir(path_group)):
            path_subject = os.path.join(os.path.join(path_general, group), dir)
            if os.path.isdir(path_subject):
                path_full_file = os.path.join(path_subject, name_file)
                time_series_aan = u.to_extract_time_series(path_full_file, atlas['AAN'])
                time_series_cortex = u.to_extract_time_series(path_full_file, atlas['Cortex'])

                cm = core.run_2_groups(time_series_aan, time_series_cortex, TR)

                connectivity_matrixs.append(cm)

                fig, ax = plt.subplots()
                plotting.plot_matrix(cm, vmax=1.0, vmin=-1.0, figure=fig)
                fig.savefig(path_group + dir + '.png', dpi=600)

        fig, ax = plt.subplots()
        cm_mean = np.mean(np.array(connectivity_matrixs), axis=0)
        plotting.plot_matrix(cm_mean, vmax=1.0, vmin=-1.0, figure=fig)
        fig.savefig(path_group + 'cm_mean.png', dpi=600)
