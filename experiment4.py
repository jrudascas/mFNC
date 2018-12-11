import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import os
import nibabel as nib
import numpy as np
from scipy import stats
from nilearn import plotting
import sys

#path_mask = '/home/runlab/data/Atlas/WM_mask_2mm.nii.gz'

path_atlas_nmi = '/home/runlab/data/Atlas/ROI_MNI_V4.nii'
path_relative_total_td_map = 'total_TD_Map.nii'
path_relative_fMRI = 'data/functional/fmriGLM.nii.gz'

path_mask = '/home/runlab/data/Atlas/WM_mask_2mm.nii.gz'
path_roi = '/home/runlab/data/Atlas/mCC.nii'
path_general = '/home/runlab/data/COMA/'

data_grey_matter = nib.load(path_mask).get_data().astype(np.int32)
data_atlas = nib.load(path_atlas_nmi).get_data()
affine_atlas = nib.load(path_atlas_nmi).affine
index_rois_atlas = np.unique(data_atlas)

print('Nmber of Index in the Atlas: ' + str(len(index_rois_atlas)))
data_mCC = nib.load(path_roi).get_data()

for group in sorted(os.listdir(path_general)):
    path_group = os.path.join(path_general, group)
    if os.path.isdir(path_group):
        matrix_by_group = []
        for subject in sorted(os.listdir(path_group)):
            path_subject = os.path.join(path_group, subject)
            if os.path.isdir(path_subject):
                print(subject)

                path_total_td_map = os.path.join(path_subject, path_relative_total_td_map)
                full_path_fMRI = os.path.join(path_subject, path_relative_fMRI)
                img_fMRI = nib.load(full_path_fMRI)
                data_fMRI = img_fMRI.get_data()
                affine_fMRI = img_fMRI.get_affine()
                list_mCC_index = []
                data_total_td_map = nib.load(path_total_td_map).get_data()
                index = 0
                for slide in range(1, data_grey_matter.shape[-1] - 1, 3):
                    for col in range(1, data_grey_matter.shape[1] - 1, 3):
                        for row in range(1, data_grey_matter.shape[0] - 1, 3):

                            indexGreyMatter = [data_grey_matter[row - 1, col - 1, slide - 1],
                                               data_grey_matter[row - 1, col, slide - 1],
                                               data_grey_matter[row - 1, col + 1, slide - 1],
                                               data_grey_matter[row, col - 1, slide - 1],
                                               data_grey_matter[row, col, slide - 1],
                                               data_grey_matter[row, col + 1, slide - 1],
                                               data_grey_matter[row + 1, col - 1, slide - 1],
                                               data_grey_matter[row + 1, col, slide - 1],
                                               data_grey_matter[row + 1, col + 1, slide - 1],
                                               data_grey_matter[row - 1, col - 1, slide],
                                               data_grey_matter[row - 1, col, slide],
                                               data_grey_matter[row - 1, col + 1, slide],
                                               data_grey_matter[row, col - 1, slide],
                                               data_grey_matter[row, col, slide],
                                               data_grey_matter[row, col + 1, slide],
                                               data_grey_matter[row + 1, col - 1, slide],
                                               data_grey_matter[row + 1, col, slide],
                                               data_grey_matter[row + 1, col + 1, slide],
                                               data_grey_matter[row - 1, col - 1, slide + 1],
                                               data_grey_matter[row - 1, col, slide + 1],
                                               data_grey_matter[row - 1, col + 1, slide + 1],
                                               data_grey_matter[row, col - 1, slide + 1],
                                               data_grey_matter[row, col, slide + 1],
                                               data_grey_matter[row, col + 1, slide + 1],
                                               data_grey_matter[row + 1, col - 1, slide + 1],
                                               data_grey_matter[row + 1, col, slide + 1],
                                               data_grey_matter[row + 1, col + 1, slide + 1]]

                            if np.count_nonzero(indexGreyMatter) > len(indexGreyMatter) * 1 / 2:

                                indexfMRI = np.array([data_fMRI[row - 1, col - 1, slide - 1],
                                                      data_fMRI[row - 1, col, slide - 1],
                                                      data_fMRI[row - 1, col + 1, slide - 1],
                                                      data_fMRI[row, col - 1, slide - 1],
                                                      data_fMRI[row, col, slide - 1],
                                                      data_fMRI[row, col + 1, slide - 1],
                                                      data_fMRI[row + 1, col - 1, slide - 1],
                                                      data_fMRI[row + 1, col, slide - 1],
                                                      data_fMRI[row + 1, col + 1, slide - 1],

                                                      data_fMRI[row - 1, col - 1, slide],
                                                      data_fMRI[row - 1, col, slide],
                                                      data_fMRI[row - 1, col + 1, slide],
                                                      data_fMRI[row, col - 1, slide],
                                                      data_fMRI[row, col, slide],
                                                      data_fMRI[row, col + 1, slide],
                                                      data_fMRI[row + 1, col - 1, slide],
                                                      data_fMRI[row + 1, col, slide],
                                                      data_fMRI[row + 1, col + 1, slide],

                                                      data_fMRI[row - 1, col - 1, slide + 1],
                                                      data_fMRI[row - 1, col, slide + 1],
                                                      data_fMRI[row - 1, col + 1, slide + 1],
                                                      data_fMRI[row, col - 1, slide + 1],
                                                      data_fMRI[row, col, slide + 1],
                                                      data_fMRI[row, col + 1, slide + 1],
                                                      data_fMRI[row + 1, col - 1, slide + 1],
                                                      data_fMRI[row + 1, col, slide + 1],
                                                      data_fMRI[row + 1, col + 1, slide + 1]])

                                if (np.mean(np.mean(indexfMRI, axis=0)) != 0):
                                    if data_mCC[row, col, slide] != 0:
                                        list_mCC_index.append(index)
                                    index += 1


                list_roi_index = []
                for roi_index in index_rois_atlas:
                    aux = np.copy(data_atlas)
                    aux[np.where(aux != roi_index)] = 0
                    list_index = []
                    index = 0
                    for slide in range(1, data_grey_matter.shape[-1] - 1, 3):
                        for col in range(1, data_grey_matter.shape[1] - 1, 3):
                            for row in range(1, data_grey_matter.shape[0] - 1, 3):

                                indexGreyMatter = [data_grey_matter[row - 1, col - 1, slide - 1],
                                                   data_grey_matter[row - 1, col, slide - 1],
                                                   data_grey_matter[row - 1, col + 1, slide - 1],
                                                   data_grey_matter[row, col - 1, slide - 1],
                                                   data_grey_matter[row, col, slide - 1],
                                                   data_grey_matter[row, col + 1, slide - 1],
                                                   data_grey_matter[row + 1, col - 1, slide - 1],
                                                   data_grey_matter[row + 1, col, slide - 1],
                                                   data_grey_matter[row + 1, col + 1, slide - 1],
                                                   data_grey_matter[row - 1, col - 1, slide],
                                                   data_grey_matter[row - 1, col, slide],
                                                   data_grey_matter[row - 1, col + 1, slide],
                                                   data_grey_matter[row, col - 1, slide],
                                                   data_grey_matter[row, col, slide],
                                                   data_grey_matter[row, col + 1, slide],
                                                   data_grey_matter[row + 1, col - 1, slide],
                                                   data_grey_matter[row + 1, col, slide],
                                                   data_grey_matter[row + 1, col + 1, slide],
                                                   data_grey_matter[row - 1, col - 1, slide + 1],
                                                   data_grey_matter[row - 1, col, slide + 1],
                                                   data_grey_matter[row - 1, col + 1, slide + 1],
                                                   data_grey_matter[row, col - 1, slide + 1],
                                                   data_grey_matter[row, col, slide + 1],
                                                   data_grey_matter[row, col + 1, slide + 1],
                                                   data_grey_matter[row + 1, col - 1, slide + 1],
                                                   data_grey_matter[row + 1, col, slide + 1],
                                                   data_grey_matter[row + 1, col + 1, slide + 1]]

                                if np.count_nonzero(indexGreyMatter) > len(indexGreyMatter) * 1 / 2:

                                    indexfMRI = np.array([data_fMRI[row - 1, col - 1, slide - 1],
                                                          data_fMRI[row - 1, col, slide - 1],
                                                          data_fMRI[row - 1, col + 1, slide - 1],
                                                          data_fMRI[row, col - 1, slide - 1],
                                                          data_fMRI[row, col, slide - 1],
                                                          data_fMRI[row, col + 1, slide - 1],
                                                          data_fMRI[row + 1, col - 1, slide - 1],
                                                          data_fMRI[row + 1, col, slide - 1],
                                                          data_fMRI[row + 1, col + 1, slide - 1],

                                                          data_fMRI[row - 1, col - 1, slide],
                                                          data_fMRI[row - 1, col, slide],
                                                          data_fMRI[row - 1, col + 1, slide],
                                                          data_fMRI[row, col - 1, slide],
                                                          data_fMRI[row, col, slide],
                                                          data_fMRI[row, col + 1, slide],
                                                          data_fMRI[row + 1, col - 1, slide],
                                                          data_fMRI[row + 1, col, slide],
                                                          data_fMRI[row + 1, col + 1, slide],

                                                          data_fMRI[row - 1, col - 1, slide + 1],
                                                          data_fMRI[row - 1, col, slide + 1],
                                                          data_fMRI[row - 1, col + 1, slide + 1],
                                                          data_fMRI[row, col - 1, slide + 1],
                                                          data_fMRI[row, col, slide + 1],
                                                          data_fMRI[row, col + 1, slide + 1],
                                                          data_fMRI[row + 1, col - 1, slide + 1],
                                                          data_fMRI[row + 1, col, slide + 1],
                                                          data_fMRI[row + 1, col + 1, slide + 1]])

                                    if (np.mean(np.mean(indexfMRI, axis=0)) != 0):
                                        if aux[row, col, slide] != 0:
                                            list_index.append(index)
                                        index += 1
                    list_roi_index.append(list_index)
                list_mean_td_mCC_to_rois = []

                for roi_index_l in list_roi_index:
                    values_list = []
                    for mCC_index in list_mCC_index:
                        for roi_index_new in roi_index_l:
                            values_list.append(data_total_td_map[mCC_index, roi_index_new])
                    list_mean_td_mCC_to_rois.append(np.mean(values_list))

                matrix_by_group.append(list_mean_td_mCC_to_rois)
        fig, ax = plt.subplots()
        plotting.plot_matrix(np.array(matrix_by_group), vmax=1.0, vmin=-1.0, figure=fig)
        fig.savefig(path_group + '/' + group + '_mCC_to_ROIs.png', dpi=600)
        nib.save(nib.Nifti1Image(np.array(matrix_by_group).astype(np.float32), affine=affine_atlas),
                 path_group + '/' + group + '_mCC_to_ROIs_raw.nii')

