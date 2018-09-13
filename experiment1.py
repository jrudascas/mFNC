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

import os

TR = 2.46
measure = 'COV'
lagged = 4
new_tr = 0.5

t1MNI = '/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'

path = '/home/runlab/data/COMA/'

dwiPath = 'data/functional/fmriGLM.nii.gz'
t1Path = 'data/structural/fwmmprage.nii.gz'
maskPath = 'data/structural/fwc1mprage.nii.gz'
timeDelayMapList = []
amplitudeWeightedTimeDelayMapList = []

print("Starting...")
for group in sorted(os.listdir(path)):
    pathInto = os.path.join(path, group)
    if os.path.isdir(pathInto):
        for dir in sorted(os.listdir(pathInto)):
            fMRI = os.path.join(os.path.join(pathInto, dir), dwiPath)
            if os.path.isdir(os.path.join(pathInto, dir)):
                print('Subject: ' + dir)

                t1 = os.path.join(os.path.join(pathInto, dir), t1Path)
                greyMatter = os.path.join(os.path.join(pathInto, dir), maskPath)

                greyMatterImg = nib.load(greyMatter)
                greyMatterData = greyMatterImg.get_data()
                greyMatterAffine = greyMatterImg.get_affine()

                fMRIImg = nib.load(fMRI)
                fMRIData = fMRIImg.get_data()
                fMRIAffine = fMRIImg.get_affine()

                t1Img = nib.load(t1)
                t1Data = t1Img.get_data()
                t1Affine = t1Img.get_affine()

                greyMatterData = (greyMatterData > 0.6)

                greyMatterData = mp.binary_closing(greyMatterData)

                timeCourse = []
                for slide in range(1, greyMatterData.shape[-1] - 1, 3):
                    for col in range(1, greyMatterData.shape[1] - 1, 3):
                        for row in range(1, greyMatterData.shape[0] - 1, 3):

                            indexGreyMatter = [greyMatterData[row-1, col-1, slide-1],
                                               greyMatterData[row-1, col, slide-1],
                                               greyMatterData[row-1, col+1, slide-1],
                                               greyMatterData[row, col-1, slide-1],
                                               greyMatterData[row, col, slide-1],
                                               greyMatterData[row, col + 1, slide-1],
                                               greyMatterData[row + 1, col-1, slide-1],
                                               greyMatterData[row + 1, col, slide-1],
                                               greyMatterData[row + 1, col+1, slide-1],
                                               greyMatterData[row-1, col-1, slide],
                                               greyMatterData[row-1, col, slide],
                                               greyMatterData[row-1, col+1, slide],
                                               greyMatterData[row, col-1, slide],
                                               greyMatterData[row, col, slide],
                                               greyMatterData[row, col + 1, slide],
                                               greyMatterData[row + 1, col-1, slide],
                                               greyMatterData[row + 1, col, slide],
                                               greyMatterData[row + 1, col+1, slide],
                                               greyMatterData[row - 1, col - 1, slide+1],
                                               greyMatterData[row - 1, col, slide+1],
                                               greyMatterData[row - 1, col + 1, slide+1],
                                               greyMatterData[row, col - 1, slide+1],
                                               greyMatterData[row, col, slide+1],
                                               greyMatterData[row, col + 1, slide+1],
                                               greyMatterData[row + 1, col - 1, slide+1],
                                               greyMatterData[row + 1, col, slide+1],
                                               greyMatterData[row + 1, col + 1, slide+1]]
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
                                indexfMRI= np.array([fMRIData[row-1, col-1, slide-1],
                                                     fMRIData[row-1, col, slide-1],
                                                     fMRIData[row-1, col+1, slide-1],
                                                     fMRIData[row, col-1, slide-1],
                                                     fMRIData[row, col, slide-1],
                                                     fMRIData[row, col + 1, slide-1],
                                                     fMRIData[row + 1, col-1, slide-1],
                                                     fMRIData[row + 1, col, slide-1],
                                                     fMRIData[row + 1, col+1, slide-1],

                                                     fMRIData[row-1, col-1, slide],
                                                     fMRIData[row-1, col, slide],
                                                     fMRIData[row-1, col+1, slide],
                                                     fMRIData[row, col-1, slide],
                                                     fMRIData[row, col, slide],
                                                     fMRIData[row, col + 1, slide],
                                                     fMRIData[row + 1, col-1, slide],
                                                     fMRIData[row + 1, col, slide],
                                                     fMRIData[row + 1, col+1, slide],

                                                     fMRIData[row - 1, col - 1, slide+1],
                                                     fMRIData[row - 1, col, slide+1],
                                                     fMRIData[row - 1, col + 1, slide+1],
                                                     fMRIData[row, col - 1, slide+1],
                                                     fMRIData[row, col, slide+1],
                                                     fMRIData[row, col + 1, slide+1],
                                                     fMRIData[row + 1, col - 1, slide+1],
                                                     fMRIData[row + 1, col, slide+1],
                                                     fMRIData[row + 1, col + 1, slide+1]])
                                if (np.mean(np.mean(indexfMRI, axis=0)) != 0):
                                    timeCourse.append(np.mean(indexfMRI, axis=0))

                print(len(timeCourse))

                FNC = f.Core()

                connectivity_matrix, td_matrix, awtd_matrix, tr = FNC.run2(np.transpose(np.array(timeCourse)), tr=TR, lag=lagged, new_tr=new_tr, measure=measure, tri_up=True)

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

                td_map = np.zeros(greyMatterData.shape)
                awtd_map = np.zeros(greyMatterData.shape)

                roiIndex = 0
                for slide in range(1, greyMatterData.shape[-1] - 1, 3):
                    for col in range(1, greyMatterData.shape[1] - 1, 3):
                        for row in range(1, greyMatterData.shape[0] - 1, 3):
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

                            indexGreyMatter = [greyMatterData[row-1, col-1, slide-1],
                                               greyMatterData[row-1, col, slide-1],
                                               greyMatterData[row-1, col+1, slide-1],
                                               greyMatterData[row, col-1, slide-1],
                                               greyMatterData[row, col, slide-1],
                                               greyMatterData[row, col + 1, slide-1],
                                               greyMatterData[row + 1, col-1, slide-1],
                                               greyMatterData[row + 1, col, slide-1],
                                               greyMatterData[row + 1, col+1, slide-1],
                                               greyMatterData[row-1, col-1, slide],
                                               greyMatterData[row-1, col, slide],
                                               greyMatterData[row-1, col+1, slide],
                                               greyMatterData[row, col-1, slide],
                                               greyMatterData[row, col, slide],
                                               greyMatterData[row, col + 1, slide],
                                               greyMatterData[row + 1, col-1, slide],
                                               greyMatterData[row + 1, col, slide],
                                               greyMatterData[row + 1, col+1, slide],
                                               greyMatterData[row - 1, col - 1, slide+1],
                                               greyMatterData[row - 1, col, slide+1],
                                               greyMatterData[row - 1, col + 1, slide+1],
                                               greyMatterData[row, col - 1, slide+1],
                                               greyMatterData[row, col, slide+1],
                                               greyMatterData[row, col + 1, slide+1],
                                               greyMatterData[row + 1, col - 1, slide+1],
                                               greyMatterData[row + 1, col, slide+1],
                                               greyMatterData[row + 1, col + 1, slide+1]]

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
                                indexfMRI= np.array([fMRIData[row-1, col-1, slide-1],
                                                     fMRIData[row-1, col, slide-1],
                                                     fMRIData[row-1, col+1, slide-1],
                                                     fMRIData[row, col-1, slide-1],
                                                     fMRIData[row, col, slide-1],
                                                     fMRIData[row, col + 1, slide-1],
                                                     fMRIData[row + 1, col-1, slide-1],
                                                     fMRIData[row + 1, col, slide-1],
                                                     fMRIData[row + 1, col+1, slide-1],

                                                     fMRIData[row-1, col-1, slide],
                                                     fMRIData[row-1, col, slide],
                                                     fMRIData[row-1, col+1, slide],
                                                     fMRIData[row, col-1, slide],
                                                     fMRIData[row, col, slide],
                                                     fMRIData[row, col + 1, slide],
                                                     fMRIData[row + 1, col-1, slide],
                                                     fMRIData[row + 1, col, slide],
                                                     fMRIData[row + 1, col+1, slide],

                                                     fMRIData[row - 1, col - 1, slide+1],
                                                     fMRIData[row - 1, col, slide+1],
                                                     fMRIData[row - 1, col + 1, slide+1],
                                                     fMRIData[row, col - 1, slide+1],
                                                     fMRIData[row, col, slide+1],
                                                     fMRIData[row, col + 1, slide+1],
                                                     fMRIData[row + 1, col - 1, slide+1],
                                                     fMRIData[row + 1, col, slide+1],
                                                     fMRIData[row + 1, col + 1, slide+1]])

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

                nib.save(nib.Nifti1Image(td_map, affine=greyMatterAffine), os.path.join(os.path.join(pathInto, dir), 'TD_Map.nii'))
                nib.save(nib.Nifti1Image(awtd_map, affine=greyMatterAffine), os.path.join(os.path.join(pathInto, dir), 'AWTD_Map.nii'))

                plotting.plot_stat_map(nib.Nifti1Image(td_map, affine=greyMatterAffine), cut_coords=[0, -28, 3], bg_img=t1MNI, vmax=0.5 * lagged, output_file=os.path.join(os.path.join(pathInto, dir), 'TD_Map.png'))
                plotting.plot_stat_map(nib.Nifti1Image(awtd_map, affine=greyMatterAffine), cut_coords=[0, -28, 3], bg_img=t1MNI, output_file=os.path.join(os.path.join(pathInto, dir), 'AWTD_Map.png'))

                timeDelayMapList.append(td_map)
                amplitudeWeightedTimeDelayMapList.append(awtd_map)

            td_matrix = np.array(timeDelayMapList)
            AWTDMatrix = np.array(amplitudeWeightedTimeDelayMapList)

            labels = range(td_matrix.shape[0])

            TDcorrelationMatrix = np.zeros((td_matrix.shape[0], td_matrix.shape[0]))
            AWTDcorrelationMatrix = np.zeros((td_matrix.shape[0], td_matrix.shape[0]))

            for indexSubject in range(td_matrix.shape[0]):
                for indexSubject2 in range(td_matrix.shape[0]):
                    #correlationMatrix[indexSubject, indexSubject2] = np.corrcoef(timeDelayMatrixs[indexSubject, :, :], timeDelayMatrixs[indexSubject2, :, :])

                    subject1FlattedTD = np.ndarray.flatten(td_matrix[indexSubject, :, :, :])
                    subject2FlattedTD = np.ndarray.flatten(td_matrix[indexSubject2, :, :, :])
                    subject1FlattedAWTD = np.ndarray.flatten(AWTDMatrix[indexSubject, :, :, :])
                    subject2FlattedAWTD = np.ndarray.flatten(AWTDMatrix[indexSubject2, :, :, :])

                    #correlationMatrix[indexSubject, indexSubject2] = dc.u_distance_correlation_sqr(subject1Flatted, subject2Flatted)
            #        correlationMatrix[indexSubject, indexSubject2] = util.mse(timeDelayMatrixs[indexSubject, :, :], timeDelayMatrixs[indexSubject2, :, :])
                    TDcorrelationMatrix[indexSubject, indexSubject2] = abs(sc.pearsonr(subject1FlattedTD, subject2FlattedTD)[0])
                    AWTDcorrelationMatrix[indexSubject, indexSubject2] = abs(sc.pearsonr(subject1FlattedAWTD, subject2FlattedAWTD)[0])

            #plott.plot_matrix(correlationMatrix, labels=labels)
            fig = drawmatrix_channels(TDcorrelationMatrix, labels, color_anchor=(0.,1.))
            fig.savefig(os.path.join(pathInto, 'TDCorrelation.png'))
            #plt.show()

            fig01 = drawmatrix_channels(AWTDcorrelationMatrix, labels, color_anchor=(0.,1.))
            fig01.savefig(os.path.join(pathInto, 'AWTDCorrelation.png'))
            #plt.show()