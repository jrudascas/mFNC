import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.ioff()

import numpy as np
import nibabel as nib
import core as f
from nilearn import plotting
import scipy.stats as sc
import scipy.ndimage.morphology as mp
from nitime.viz import drawmatrix_channels
import timeit

import os

TR = 2.46
measure = 'COV'
lagged = 4
new_tr = None

path_study = '/home/runlab/data/COMA/'
path_relative_fMRI = 'data/functional/fmriGLM.nii.gz'

path_mask = '/home/runlab/data/Atlas/WM_mask_2mm.nii.gz'


img_grey_matter = nib.load(path_mask)
data_grey_matter = img_grey_matter.get_data().astype(np.int8)
affine_grey_matter = img_grey_matter.get_affine()

#data_grey_matter = (data_grey_matter > 0.9)
#data_grey_matter = mp.binary_closing(data_grey_matter)

td_map_list = []
awtd_map_list = []

for group in sorted(os.listdir(path_study)):
    path_group = os.path.join(path_study, group)
    if os.path.isdir(path_group):
        for dir in sorted(os.listdir(path_group)):
            path_subject = os.path.join(path_group, dir)
            if os.path.isdir(path_subject):
                print('Subject: ' + dir)
                full_path_fMRI = os.path.join(path_subject, path_relative_fMRI)

                img_fMRI = nib.load(full_path_fMRI)
                data_fMRI = img_fMRI.get_data()
                affine_fMRI = img_fMRI.get_affine()

                time_courses = []
                for slide in range(1, data_grey_matter.shape[-1] - 1, 2):
                    for col in range(1, data_grey_matter.shape[1] - 1, 2):
                        for row in range(1, data_grey_matter.shape[0] - 1, 2):

                            indexGreyMatter = [data_grey_matter[row, col, slide],
                                               data_grey_matter[row, col + 1, slide],
                                               data_grey_matter[row + 1, col, slide],
                                               data_grey_matter[row + 1, col + 1, slide],
                                               data_grey_matter[row, col, slide + 1],
                                               data_grey_matter[row, col + 1, slide + 1],
                                               data_grey_matter[row + 1, col, slide + 1],
                                               data_grey_matter[row + 1, col + 1, slide + 1]]

                            if np.count_nonzero(indexGreyMatter) > len(
                                    indexGreyMatter) * 1 / 2:  # more than 50% of voxel are into the grey matter

                                indexfMRI = np.array([data_fMRI[row, col, slide],
                                                      data_fMRI[row, col + 1, slide],
                                                      data_fMRI[row + 1, col, slide],
                                                      data_fMRI[row + 1, col + 1, slide],
                                                      data_fMRI[row, col, slide + 1],
                                                      data_fMRI[row, col + 1, slide + 1],
                                                      data_fMRI[row + 1, col, slide + 1],
                                                      data_fMRI[row + 1, col + 1, slide + 1]])

                                if (np.mean(np.mean(indexfMRI, axis=0)) != 0):
                                    time_courses.append(np.mean(indexfMRI, axis=0))

                c = f.Core()

                print(len(time_courses))
                connectivity_matrix, td_matrix, awtd_matrix, tr = c.run2(np.transpose(np.array(time_courses)), tr=TR,
                                                                         lag=lagged, new_tr=new_tr, measure=measure,
                                                                         tri_up=True)

                td_matrix = td_matrix * tr
                awtd_matrix = awtd_matrix * tr

                upperTD = np.copy(td_matrix)
                lowerTD = np.transpose(np.copy(td_matrix))

                td_matrix_total = upperTD - lowerTD
                td_projection = np.mean(td_matrix_total, axis=0)

                upperAWTD = np.copy(awtd_matrix)
                lowerAWTD = np.transpose(np.copy(awtd_matrix))

                awtd_matrix_total = upperAWTD - lowerAWTD
                awtd_projection = np.mean(awtd_matrix_total, axis=0)

                td_map = np.zeros(data_grey_matter.shape)
                awtd_map = np.zeros(data_grey_matter.shape)

                roiIndex = 0
                for slide in range(1, data_grey_matter.shape[-1] - 1, 2):
                    for col in range(1, data_grey_matter.shape[1] - 1, 2):
                        for row in range(1, data_grey_matter.shape[0] - 1, 2):

                            indexGreyMatter = [data_grey_matter[row, col, slide],
                                               data_grey_matter[row, col + 1, slide],
                                               data_grey_matter[row + 1, col, slide],
                                               data_grey_matter[row + 1, col + 1, slide],
                                               data_grey_matter[row, col, slide + 1],
                                               data_grey_matter[row, col + 1, slide + 1],
                                               data_grey_matter[row + 1, col, slide + 1],
                                               data_grey_matter[row + 1, col + 1, slide + 1]]

                            if np.count_nonzero(indexGreyMatter) > len(
                                    indexGreyMatter) * 1 / 2:  # more than 50% of voxel are into the grey matter

                                """
                                timeDelayMap[row, col, slide] = timeDelayProjection[roiIndex]
                                timeDelayMap[row, col + 1, slide] = timeDelayProjection[roiIndex]
                                timeDelayMap[row + 1, col, slide] = timeDelayProjection[roiIndex]
                                timeDelayMap[row + 1, col + 1, slide] = timeDelayProjection[roiIndex]
                                timeDelayMap[row, col, slide + 1] = timeDelayProjection[roiIndex]
                                timeDelayMap[row, col + 1, slide + 1] = timeDelayProjection[roiIndex]
                                timeDelayMap[row + 1, col, slide + 1] = timeDelayProjection[roiIndex]
                                timeDelayMap[row + 1, col + 1, slide + 1] = timeDelayProjection[roiIndex]

                                """

                                indexfMRI = np.array([data_fMRI[row, col, slide],
                                                     data_fMRI[row, col + 1, slide],
                                                     data_fMRI[row + 1, col, slide],
                                                     data_fMRI[row + 1, col + 1, slide],
                                                     data_fMRI[row, col, slide + 1],
                                                     data_fMRI[row, col + 1, slide + 1],
                                                     data_fMRI[row + 1, col, slide + 1],
                                                     data_fMRI[row + 1, col + 1, slide + 1]])

                                if (np.mean(np.mean(indexfMRI, axis=0)) != 0):

                                    td_map[row, col, slide] = td_projection[roiIndex]
                                    td_map[row, col + 1, slide] = td_projection[roiIndex]
                                    td_map[row + 1, col, slide] = td_projection[roiIndex]
                                    td_map[row + 1, col + 1, slide] = td_projection[roiIndex]
                                    td_map[row, col, slide + 1] = td_projection[roiIndex]
                                    td_map[row, col + 1, slide + 1] = td_projection[roiIndex]
                                    td_map[row + 1, col, slide + 1] = td_projection[roiIndex]
                                    td_map[row + 1, col + 1, slide + 1] = td_projection[roiIndex]

                                    awtd_map[row, col, slide] = awtd_projection[roiIndex]
                                    awtd_map[row, col + 1, slide] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col, slide] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col + 1, slide] = awtd_projection[roiIndex]
                                    awtd_map[row, col, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row, col + 1, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col + 1, slide + 1] = awtd_projection[roiIndex]

                                    roiIndex = roiIndex + 1

                nib.save(nib.Nifti1Image(td_map, affine=affine_grey_matter),
                         os.path.join(os.path.join(path_group, dir), 'TD_Map.nii'))
                nib.save(nib.Nifti1Image(awtd_map, affine=affine_grey_matter),
                         os.path.join(os.path.join(path_group, dir), 'AWTD_Map.nii'))

                plotting.plot_stat_map(nib.Nifti1Image(td_map, affine=affine_grey_matter), cut_coords=[0, -28, 3],
                                       vmax=0.5 * lagged,
                                       output_file=os.path.join(os.path.join(path_group, dir), 'TD_Map.png'))
                plotting.plot_stat_map(nib.Nifti1Image(awtd_map, affine=affine_grey_matter), cut_coords=[0, -28, 3],
                                       vmax=0.5 * lagged,
                                       output_file=os.path.join(os.path.join(path_group, dir), 'AWTD_Map.png'))

                td_map_list.append(td_map)
                awtd_map_list.append(awtd_map)

            td_matrixs = np.array(td_map_list)
            awtd_matrixs = np.array(awtd_map_list)

            labels = range(td_matrixs.shape[0])

            td_correlation_matrix = np.zeros((td_matrixs.shape[0], td_matrixs.shape[0]))
            awtd_correlation_matrix = np.zeros((td_matrixs.shape[0], td_matrixs.shape[0]))

            for index1 in range(td_matrixs.shape[0]):
                for index2 in range(td_matrixs.shape[0]):
                    td_flatted_1 = np.ndarray.flatten(td_matrixs[index1, :, :, :])
                    td_flatted_2 = np.ndarray.flatten(td_matrixs[index2, :, :, :])
                    awtd_flatted_1 = np.ndarray.flatten(awtd_matrixs[index1, :, :, :])
                    awtd_flatted_2 = np.ndarray.flatten(awtd_matrixs[index2, :, :, :])

                    td_correlation_matrix[index1, index2] = abs(sc.pearsonr(td_flatted_1, td_flatted_2)[0])
                    awtd_correlation_matrix[index1, index2] = abs(sc.pearsonr(awtd_flatted_1, awtd_flatted_2)[0])

            fig = drawmatrix_channels(td_correlation_matrix, labels, color_anchor=(0., 1.))
            fig.savefig(os.path.join(path_group, 'TDCorrelation.png'))

            fig01 = drawmatrix_channels(awtd_correlation_matrix, labels, color_anchor=(0., 1.))
            fig01.savefig(os.path.join(path_group, 'AWTDCorrelation.png'))

'''

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import nibabel as nib
import core as f
from nilearn import plotting
import scipy.stats as sc
import scipy.ndimage.morphology as mp
from nitime.viz import drawmatrix_channels
import timeit

import os

TR = 2.46
measure = 'COV'
lagged = 4
new_tr = 0.5

path_study = '/home/runlab/data/COMA/'
path_relative_fMRI = 'data/functional/fmriGLM.nii.gz'

path_mask = path_study + 'HC/Cahodessur/data/structural/fwc1mprage.nii.gz'

img_grey_matter = nib.load(path_mask)
data_grey_matter = img_grey_matter.get_data()
affine_grey_matter = img_grey_matter.get_affine()

data_grey_matter = (data_grey_matter > 0.4)
data_grey_matter = mp.binary_closing(data_grey_matter)

td_map_list = []
awtd_map_list = []

for group in sorted(os.listdir(path_study)):
    path_group = os.path.join(path_study, group)
    if os.path.isdir(path_group):
        for dir in sorted(os.listdir(path_group)):
            path_subject = os.path.join(path_group, dir)
            if os.path.isdir(path_subject):
                print('Subject: ' + dir)
                full_path_fMRI = os.path.join(path_subject, path_relative_fMRI)

                img_fMRI = nib.load(full_path_fMRI)
                data_fMRI = img_fMRI.get_data()
                affine_fMRI = img_fMRI.get_affine()

                time_courses = []
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
                            """
                
                            indexGreyMatter = [greyMatterData[row, col, slide],
                                               greyMatterData[row, col + 1, slide],
                                               greyMatterData[row + 1, col, slide],
                                               greyMatterData[row + 1, col + 1, slide],
                                               greyMatterData[row, col, slide + 1],
                                               greyMatterData[row, col + 1, slide + 1],
                                               greyMatterData[row + 1, col, slide + 1],
                                               greyMatterData[row + 1, col + 1, slide + 1]]
                            """
                            if np.count_nonzero(indexGreyMatter) > len(indexGreyMatter)*1/2: # more than 50% of voxel are into the grey matter
                                """
                                indexfMRI = np.array([fMRIData[row, col, slide],
                                                      fMRIData[row, col + 1, slide],
                                                      fMRIData[row + 1, col, slide],
                                                      fMRIData[row + 1, col + 1, slide],
                                                      fMRIData[row, col, slide + 1],
                                                      fMRIData[row, col + 1, slide + 1],
                                                      fMRIData[row + 1, col, slide + 1],
                                                      fMRIData[row + 1, col + 1, slide + 1]])
                                """
                                indexfMRI= np.array([data_fMRI[row - 1, col - 1, slide - 1],
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
                                    time_courses.append(np.mean(indexfMRI, axis=0))

                c = f.Core()

                connectivity_matrix, td_matrix, awtd_matrix, tr = c.run2(np.transpose(np.array(time_courses)), tr=TR, lag=lagged, new_tr=new_tr, measure=measure, tri_up=True)

                td_matrix = td_matrix * tr
                awtd_matrix = awtd_matrix * tr

                upperTD = np.copy(td_matrix)
                lowerTD = np.transpose(np.copy(td_matrix))

                td_matrix_total = upperTD - lowerTD
                td_projection = np.mean(td_matrix_total, axis=0)

                upperAWTD = np.copy(awtd_matrix)
                lowerAWTD = np.transpose(np.copy(awtd_matrix))

                awtd_matrix_total = upperAWTD - lowerAWTD
                awtd_projection = np.mean(awtd_matrix_total, axis=0)

                td_map = np.zeros(data_grey_matter.shape)
                awtd_map = np.zeros(data_grey_matter.shape)

                roiIndex = 0
                for slide in range(1, data_grey_matter.shape[-1] - 1, 3):
                    for col in range(1, data_grey_matter.shape[1] - 1, 3):
                        for row in range(1, data_grey_matter.shape[0] - 1, 3):
                            """
                            indexGreyMatter = [greyMatterData[row, col, slide],
                                               greyMatterData[row, col + 1, slide],
                                               greyMatterData[row + 1, col, slide],
                                               greyMatterData[row + 1, col + 1, slide],
                                               greyMatterData[row, col, slide + 1],
                                               greyMatterData[row, col + 1, slide + 1],
                                               greyMatterData[row + 1, col, slide + 1],
                                               greyMatterData[row + 1, col + 1, slide + 1]]
                            """

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

                            if np.count_nonzero(indexGreyMatter) > len(indexGreyMatter)*1/2: # more than 50% of voxel are into the grey matter

                                """
                                timeDelayMap[row, col, slide] = timeDelayProjection[roiIndex]
                                timeDelayMap[row, col + 1, slide] = timeDelayProjection[roiIndex]
                                timeDelayMap[row + 1, col, slide] = timeDelayProjection[roiIndex]
                                timeDelayMap[row + 1, col + 1, slide] = timeDelayProjection[roiIndex]
                                timeDelayMap[row, col, slide + 1] = timeDelayProjection[roiIndex]
                                timeDelayMap[row, col + 1, slide + 1] = timeDelayProjection[roiIndex]
                                timeDelayMap[row + 1, col, slide + 1] = timeDelayProjection[roiIndex]
                                timeDelayMap[row + 1, col + 1, slide + 1] = timeDelayProjection[roiIndex]
                
                                """
                                indexfMRI= np.array([data_fMRI[row - 1, col - 1, slide - 1],
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

                                    td_map[row - 1, col - 1, slide - 1] = td_projection[roiIndex]
                                    td_map[row - 1, col, slide - 1] = td_projection[roiIndex]
                                    td_map[row - 1, col + 1, slide - 1] = td_projection[roiIndex]
                                    td_map[row, col - 1, slide - 1] = td_projection[roiIndex]
                                    td_map[row, col, slide - 1] = td_projection[roiIndex]
                                    td_map[row, col + 1, slide - 1] = td_projection[roiIndex]
                                    td_map[row + 1, col - 1, slide - 1] = td_projection[roiIndex]
                                    td_map[row + 1, col, slide - 1] = td_projection[roiIndex]
                                    td_map[row + 1, col + 1, slide - 1] = td_projection[roiIndex]
                                    td_map[row - 1, col - 1, slide] = td_projection[roiIndex]
                                    td_map[row - 1, col, slide] = td_projection[roiIndex]
                                    td_map[row - 1, col + 1, slide] = td_projection[roiIndex]
                                    td_map[row, col - 1, slide] = td_projection[roiIndex]
                                    td_map[row, col, slide] = td_projection[roiIndex]
                                    td_map[row, col + 1, slide] = td_projection[roiIndex]
                                    td_map[row + 1, col - 1, slide] = td_projection[roiIndex]
                                    td_map[row + 1, col, slide] = td_projection[roiIndex]
                                    td_map[row + 1, col + 1, slide] = td_projection[roiIndex]
                                    td_map[row - 1, col - 1, slide + 1] = td_projection[roiIndex]
                                    td_map[row - 1, col, slide + 1] = td_projection[roiIndex]
                                    td_map[row - 1, col + 1, slide + 1] = td_projection[roiIndex]
                                    td_map[row, col - 1, slide + 1] = td_projection[roiIndex]
                                    td_map[row, col, slide + 1] = td_projection[roiIndex]
                                    td_map[row, col + 1, slide + 1] = td_projection[roiIndex]
                                    td_map[row + 1, col - 1, slide + 1] = td_projection[roiIndex]
                                    td_map[row + 1, col, slide + 1] = td_projection[roiIndex]
                                    td_map[row + 1, col + 1, slide + 1] = td_projection[roiIndex]

                                    awtd_map[row - 1, col - 1, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row - 1, col, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row - 1, col + 1, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row, col - 1, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row, col, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row, col + 1, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col - 1, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col + 1, slide - 1] = awtd_projection[roiIndex]
                                    awtd_map[row - 1, col - 1, slide] = awtd_projection[roiIndex]
                                    awtd_map[row - 1, col, slide] = awtd_projection[roiIndex]
                                    awtd_map[row - 1, col + 1, slide] = awtd_projection[roiIndex]
                                    awtd_map[row, col - 1, slide] = awtd_projection[roiIndex]
                                    awtd_map[row, col, slide] = awtd_projection[roiIndex]
                                    awtd_map[row, col + 1, slide] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col - 1, slide] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col, slide] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col + 1, slide] = awtd_projection[roiIndex]
                                    awtd_map[row - 1, col - 1, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row - 1, col, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row - 1, col + 1, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row, col - 1, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row, col, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row, col + 1, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col - 1, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col, slide + 1] = awtd_projection[roiIndex]
                                    awtd_map[row + 1, col + 1, slide + 1] = awtd_projection[roiIndex]

                                    roiIndex = roiIndex + 1

                nib.save(nib.Nifti1Image(td_map, affine=affine_grey_matter), os.path.join(os.path.join(path_group, dir), 'TD_Map.nii'))
                nib.save(nib.Nifti1Image(awtd_map, affine=affine_grey_matter), os.path.join(os.path.join(path_group, dir), 'AWTD_Map.nii'))

                plotting.plot_stat_map(nib.Nifti1Image(td_map, affine=affine_grey_matter), cut_coords=[0, -28, 3], vmax=0.5 * lagged, output_file=os.path.join(os.path.join(path_group, dir), 'TD_Map.png'))
                plotting.plot_stat_map(nib.Nifti1Image(awtd_map, affine=affine_grey_matter), cut_coords=[0, -28, 3], vmax=0.5 * lagged, output_file=os.path.join(os.path.join(path_group, dir), 'AWTD_Map.png'))

                td_map_list.append(td_map)
                awtd_map_list.append(awtd_map)

            td_matrixs = np.array(td_map_list)
            awtd_matrixs = np.array(awtd_map_list)

            labels = range(td_matrixs.shape[0])

            td_correlation_matrix = np.zeros((td_matrixs.shape[0], td_matrixs.shape[0]))
            awtd_correlation_matrix = np.zeros((td_matrixs.shape[0], td_matrixs.shape[0]))

            for index1 in range(td_matrixs.shape[0]):
                for index2 in range(td_matrixs.shape[0]):
                    td_flatted_1 = np.ndarray.flatten(td_matrixs[index1, :, :, :])
                    td_flatted_2 = np.ndarray.flatten(td_matrixs[index2, :, :, :])
                    awtd_flatted_1 = np.ndarray.flatten(awtd_matrixs[index1, :, :, :])
                    awtd_flatted_2 = np.ndarray.flatten(awtd_matrixs[index2, :, :, :])

                    td_correlation_matrix[index1, index2] = abs(sc.pearsonr(td_flatted_1, td_flatted_2)[0])
                    awtd_correlation_matrix[index1, index2] = abs(sc.pearsonr(awtd_flatted_1, awtd_flatted_2)[0])

            fig = drawmatrix_channels(td_correlation_matrix, labels, color_anchor=(0., 1.))
            fig.savefig(os.path.join(path_group, 'TDCorrelation.png'))

            fig01 = drawmatrix_channels(awtd_correlation_matrix, labels, color_anchor=(0., 1.))
            fig01.savefig(os.path.join(path_group, 'AWTDCorrelation.png'))

'''