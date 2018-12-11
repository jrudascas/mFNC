import numpy as np

import matplotlib.pyplot as plt
#plt.ioff()

import nibabel as nib
import os
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from nistats.utils import z_score
from scipy.stats import ttest_1samp, ttest_ind, f_oneway, mannwhitneyu
from nistats.thresholding import map_threshold
import sys
import utils as u
from nilearn import plotting

path_general = '/home/jrudascas/Desktop/Test/NewTest/'
name_td_map = 'TD_Map.nii'
path_mask = '/home/jrudascas/Desktop/DWITest/Additionals/Standards/WM_mask_2mm.nii.gz'

dict_td_maps = {}
dict_td_maps_new = {}

for group in sorted(os.listdir(path_general)):
    if os.path.isdir(os.path.join(path_general, group)):
        path_within_group = os.path.join(path_general, group)
        td_maps = []
        aux = np.zeros((91,109,91,len(sorted(os.listdir(path_within_group)))))
        cont = 0
        for dir in sorted(os.listdir(path_within_group)):
            if os.path.isdir(os.path.join(path_within_group, dir)):
                path_td_map = os.path.join(os.path.join(path_within_group, dir), name_td_map)
                td_maps.append(nib.load(path_td_map).get_data())

                aux[:,:,:, cont] = nib.load(path_td_map).get_data()
                cont = cont + 1

        dict_td_maps[group] = np.array(td_maps)
        dict_td_maps_new[group] = aux

affine = nib.load(path_td_map).affine
group = 'HC'
differences_hc_mcs = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))
differences_hc_uws = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))
differences_mcs_uws = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))

print(dict_td_maps[group].shape)

tmap_hc  = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))
tmap_mcs = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))
tmap_uws = np.zeros((dict_td_maps[group].shape[1], dict_td_maps[group].shape[2], dict_td_maps[group].shape[3]))

'''
for dim1 in range(dict_td_maps[group].shape[1]):
    for dim2 in range(dict_td_maps[group].shape[2]):
        for dim3 in range(dict_td_maps[group].shape[3]):
            if np.sum(dict_td_maps['HC'][:, dim1, dim2, dim3]) != 0:
                t, p = ttest_1samp(dict_td_maps['HC'][:, dim1, dim2, dim3], 0)
                tmap_hc[dim1, dim2, dim3] = stats.norm.isf(p)

            if np.sum(dict_td_maps['MCS'][:, dim1, dim2, dim3]) != 0:
                t, p = ttest_1samp(dict_td_maps['MCS'][:, dim1, dim2, dim3], 0)
                tmap_mcs[dim1, dim2, dim3] = stats.norm.isf(p)

            if np.sum(dict_td_maps['UWS'][:, dim1, dim2, dim3]) != 0:
                t, p = ttest_1samp(dict_td_maps['UWS'][:, dim1, dim2, dim3], 0)
                tmap_uws[dim1, dim2, dim3] = stats.norm.isf(p)

nib.save(nib.Nifti1Image(tmap_hc, affine=affine), os.path.join(path_general, 'new_z-map_hc2.nii'))
nib.save(nib.Nifti1Image(tmap_mcs, affine=affine), os.path.join(path_general, 'new_z-map_mcs2.nii'))
nib.save(nib.Nifti1Image(tmap_uws, affine=affine), os.path.join(path_general, 'new_z-map_uws2.nii'))

print("Max:" + str(u.absmax(tmap_uws)))

thresholded_map2, threshold2 = map_threshold(nib.Nifti1Image(tmap_mcs, affine=affine), mask_img=path_mask, level=0.0083, height_control='fpr', cluster_threshold=160)
thresholded_map1, threshold1 = map_threshold(nib.Nifti1Image(tmap_hc, affine=affine), mask_img=path_mask, level=0.0083, height_control='fdr', cluster_threshold=160)
thresholded_map3, threshold3 = map_threshold(nib.Nifti1Image(tmap_uws, affine=affine), mask_img=path_mask, level=0.0083, height_control='fpr', cluster_threshold=160)

print(threshold1)
print(threshold2)
print(threshold3)

nib.save(thresholded_map1, os.path.join(path_general, 'new_z-map_hc.nii'))
nib.save(thresholded_map2, os.path.join(path_general, 'new_z-map_mcs.nii'))
nib.save(thresholded_map3, os.path.join(path_general, 'new_z-map_uws.nii'))

sys.exit(0)
'''
for dim1 in range(dict_td_maps[group].shape[1]):
    for dim2 in range(dict_td_maps[group].shape[2]):
        for dim3 in range(dict_td_maps[group].shape[3]):
            if np.sum(dict_td_maps['HC'][:, dim1, dim2, dim3]) != 0 and np.sum(dict_td_maps['MCS'][:, dim1, dim2, dim3]) != 0:
                #t, p = stats.mannwhitneyu(dict_td_maps['HC'][dict_td_maps['HC'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3], dict_td_maps['MCS'][dict_td_maps['MCS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3])
                t, p = mannwhitneyu(
                    dict_td_maps['HC'][dict_td_maps['HC'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3],
                    dict_td_maps['MCS'][dict_td_maps['MCS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3])

                differences_hc_mcs[dim1, dim2, dim3] = stats.norm.isf(p)

            if np.sum(dict_td_maps['HC'][:, dim1, dim2, dim3]) != 0 and np.sum(dict_td_maps['UWS'][:, dim1, dim2, dim3]) != 0:
                t, p = mannwhitneyu(
                    dict_td_maps['HC'][dict_td_maps['HC'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3],
                    dict_td_maps['UWS'][dict_td_maps['UWS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3])

                differences_hc_uws[dim1, dim2, dim3] = stats.norm.isf(p)

            if np.sum(dict_td_maps['MCS'][:, dim1, dim2, dim3]) != 0 and np.sum(dict_td_maps['UWS'][:, dim1, dim2, dim3]) != 0:
                t, p = mannwhitneyu(
                    dict_td_maps['MCS'][dict_td_maps['MCS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3],
                    dict_td_maps['UWS'][dict_td_maps['UWS'][:, dim1, dim2, dim3] != 0, dim1, dim2, dim3])

                differences_mcs_uws[dim1, dim2, dim3] = stats.norm.isf(p)

nib.save(nib.Nifti1Image(dict_td_maps['HC'], affine=affine), os.path.join(path_general, 'hc.nii'))
nib.save(nib.Nifti1Image(dict_td_maps['MCS'], affine=affine), os.path.join(path_general, 'mcs.nii'))
nib.save(nib.Nifti1Image(dict_td_maps['UWS'], affine=affine), os.path.join(path_general, 'uws.nii'))

z_map_hc_mcs = nib.Nifti1Image(differences_hc_mcs, affine=affine)
z_map_hc_uws = nib.Nifti1Image(differences_hc_uws, affine=affine)
z_map_mcs_uws = nib.Nifti1Image(differences_mcs_uws, affine=affine)

thresholded_map1, threshold1 = map_threshold(z_map_hc_mcs, level=.05, height_control='fpr', cluster_threshold=200)
thresholded_map2, threshold2 = map_threshold(z_map_hc_uws, level=.05, height_control='fpr', cluster_threshold=200)
thresholded_map3, threshold3 = map_threshold(z_map_mcs_uws, level=.05, height_control='fpr', cluster_threshold=200)

#thresholded_map2, threshold2 = map_threshold(z_map_hc_mcs, level=.05, height_control='fdr')

#display = plotting.plot_stat_map(z_map_hc_mcs, title='Raw z map')

#plotting.show()

#plotting.plot_stat_map(z_map_hc_mcs, cut_coords=display.cut_coords, threshold=threshold1,
#                       title='Thresholded z map, fpr <.05, clusters > 10 voxels')

#plotting.show()

#plotting.plot_stat_map(thresholded_map2, cut_coords=display.cut_coords, threshold=threshold2,
#                       title='Thresholded z map, expected fdr = .05')

#plotting.show()

nib.save(thresholded_map1, os.path.join(path_general, 'z-map_td_hc_mcs.nii'))
nib.save(thresholded_map2, os.path.join(path_general, 'z-map_td_hc_uws.nii'))
nib.save(thresholded_map3, os.path.join(path_general, 'z-map_td_mcs_uws.nii'))

