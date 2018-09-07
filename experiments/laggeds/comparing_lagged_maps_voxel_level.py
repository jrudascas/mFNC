import numpy as np

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
plt.ioff()

import nibabel as nib
import os
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import utils as u
from nilearn import plotting

path_general = '/home/jrudascas/Desktop/Test/'
name_td_map = 'TD_Map.nii'

dict_td_maps = {}
for group in sorted(os.listdir(path_general)):
    if os.path.isdir(os.path.join(path_general, group)):
        path_within_group = os.path.join(path_general, group)
        td_maps = []

        for dir in sorted(os.listdir(path_within_group)):
            if os.path.isdir(os.path.join(path_within_group, dir)):
                path_td_map = os.path.join(os.path.join(path_within_group, dir), name_td_map)

                td_maps.append(nib.load(path_td_map).get_data())

        dict_td_maps[group] = np.array(td_maps)

affine = nib.load(path_td_map).affine
group = 'HC'
differences_hc_mcs = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))
differences_hc_uws = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))
differences_mcs_uws = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))

e1 = 0
e2 = 1
s1 = -7
s2 = 7

for dim1 in range(dict_td_maps[group].shape[1]):
    for dim2 in range(dict_td_maps[group].shape[2]):
        for dim3 in range(dict_td_maps[group].shape[3]):
            if np.sum(dict_td_maps['HC'][:, dim1, dim2, dim3]) != 0 and np.sum(dict_td_maps['MCS'][:, dim1, dim2, dim3]) != 0:
                t, p = stats.mannwhitneyu(dict_td_maps['HC'][dict_td_maps['HC'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3], dict_td_maps['MCS'][dict_td_maps['MCS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3])
                differences_hc_mcs[dim1, dim2, dim3] = u.to_project_interval(1-p,e1, e2, s1, s2)
            else:
                differences_hc_mcs[dim1, dim2, dim3] = s1

            if np.sum(dict_td_maps['HC'][:, dim1, dim2, dim3]) != 0 and np.sum(dict_td_maps['UWS'][:, dim1, dim2, dim3]) != 0:
                t, p = stats.mannwhitneyu(dict_td_maps['HC'][dict_td_maps['HC'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3], dict_td_maps['UWS'][dict_td_maps['UWS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3])
                differences_hc_uws[dim1, dim2, dim3] = u.to_project_interval(1-p,e1, e2, s1, s2)
            else:
                differences_hc_uws[dim1, dim2, dim3] = s1

            if np.sum(dict_td_maps['MCS'][:, dim1, dim2, dim3]) != 0 and np.sum(dict_td_maps['UWS'][:, dim1, dim2, dim3]) != 0:
                t, p = stats.mannwhitneyu(dict_td_maps['MCS'][dict_td_maps['MCS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3], dict_td_maps['UWS'][dict_td_maps['UWS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3])
                differences_mcs_uws[dim1, dim2, dim3] = u.to_project_interval(1-p,e1, e2, s1, s2)
            else:
                differences_mcs_uws[dim1, dim2, dim3] = s1

nib.save(nib.Nifti1Image(differences_hc_mcs, affine=affine), os.path.join(path_general, 'differences_hc_mcs.nii'))
nib.save(nib.Nifti1Image(differences_hc_uws, affine=affine), os.path.join(path_general, 'differences_hc_uws.nii'))
nib.save(nib.Nifti1Image(differences_mcs_uws, affine=affine), os.path.join(path_general, 'differences_mcs_uws.nii'))