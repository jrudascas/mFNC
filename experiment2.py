import numpy as np

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
plt.ioff()

import nibabel as nib
from nilearn import plotting
from sklearn.metrics import mean_squared_error, r2_score
import utils as ut
import os
import warnings
warnings.filterwarnings("ignore")

t1MNI = '/home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T1_2mm_brain.nii.gz'
atlas = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'

path = '/home/jrudascas/Desktop/Test/NewTest/'
TDMapName = 'TD_Map.nii'
awtd_map_relative_path = 'AWTD_Map.nii'
maskPath = 'data/structural/fwc1mprage.nii.gz'

atlasImg = nib.load(atlas)
atlasData = atlasImg.get_data()
atlasAffine = atlasImg.get_affine()
groupTDMap = []

indexROI = np.unique(atlasData)

for group in sorted(os.listdir(path)):
    if os.path.isdir(os.path.join(path, group)):
        pathInto = os.path.join(path, group)
        TDMap = []
        td_map_mean = np.zeros(atlasData.shape)

        awtd_map = []
        awtd_mean_map = np.zeros(atlasData.shape)

        cont = 0
        for dir in sorted(os.listdir(pathInto)):
            if os.path.isdir(os.path.join(os.path.join(path, group), dir)):
                print(os.path.join(os.path.join(path, group), dir))
                cont = cont + 1
                TDMapPath = os.path.join(os.path.join(pathInto, dir), TDMapName)
                path_awtd_map = os.path.join(os.path.join(pathInto, dir), awtd_map_relative_path)

                img_td_map = nib.load(TDMapPath)
                data_td_map = img_td_map.get_data()
                affine_td_map = img_td_map.get_affine()

                img_awtd_map = nib.load(path_awtd_map)
                data_awtd_map = img_awtd_map.get_data()
                affine_awtd_map = img_awtd_map.get_affine()

                roiLags = []
                for roi in indexROI:
                    roiLags.append(np.mean(data_td_map[atlasData == roi]))
                TDMap.append(roiLags)

                td_map_mean = data_td_map + td_map_mean
                awtd_mean_map = data_awtd_map + awtd_mean_map

        td_map_mean = td_map_mean / cont
        awtd_mean_map = awtd_mean_map/cont

        nib.save(nib.Nifti1Image(td_map_mean, affine=affine_td_map), os.path.join(pathInto, 'meanTDMap_' + group + '.nii'))
        nib.save(nib.Nifti1Image(awtd_mean_map, affine=affine_awtd_map), os.path.join(pathInto, 'meanAWTDMap_' + group + '.nii'))

        plotting.plot_stat_map(nib.Nifti1Image(td_map_mean, affine=affine_td_map), cut_coords=[0, -28, 3], bg_img=t1MNI,
                               vmax=0.5 * 4, output_file=os.path.join(pathInto, 'meanTDMap_' + group + '.png'))
        #plt.show()

        groupTDMap.append(TDMap)

#meanTDMAP = np.asarray(groupTDMap)

ut.toFindStatisticDifference(np.asarray(groupTDMap[0]), np.asarray(groupTDMap[1]), threshold=0.05)
ut.toFindStatisticDifference(np.asarray(groupTDMap[0]), np.asarray(groupTDMap[2]), threshold=0.05)
ut.toFindStatisticDifference(np.asarray(groupTDMap[1]), np.asarray(groupTDMap[2]), threshold=0.05)

regr = linear_model.LinearRegression()

listRoiTo = [8, 20,36]


for roiTo in listRoiTo:
    # Train the model using the training sets
    x_train = np.linspace(1, len(groupTDMap[0]) + len(groupTDMap[1]) + len(groupTDMap[2]), len(groupTDMap[0]) + len(groupTDMap[1]) + len(groupTDMap[2]))

    y_train = np.concatenate((np.asarray(groupTDMap[0])[:,roiTo], np.asarray(groupTDMap[1])[:,roiTo]), axis=0)
    y_train = np.concatenate((y_train, np.asarray(groupTDMap[2])[:,roiTo]), axis=0)

    x_test = x_train

    regr.fit(np.transpose(np.matrix(x_train)), np.transpose(np.matrix(y_train)))

    y_pred = regr.predict(np.transpose(np.matrix(x_test)))

    plt.plot(x_test, y_pred, color='blue', linewidth=3)

    print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
    print('R2: %.2f' % r2_score(y_train, y_pred))

    plt.scatter(range(0,len(groupTDMap[0])), np.asarray(groupTDMap[0])[:,roiTo], alpha=0.5)
    plt.scatter(range(len(groupTDMap[0]), len(groupTDMap[0]) + len(groupTDMap[1])), np.asarray(groupTDMap[1])[:,roiTo], alpha=0.5)
    plt.scatter(range(len(groupTDMap[0]) + len(groupTDMap[1]), len(groupTDMap[0]) + len(groupTDMap[1]) + len(groupTDMap[2])), np.asarray(groupTDMap[2])[:,roiTo], alpha=0.5)
    plt.title('ROI: ' + str(roiTo) + ' - ' + 'R2: ' + str(r2_score(y_train, y_pred)))
    plt.show()

display = plotting.plot_anat(t1MNI, cut_coords=[0,-60,45])

display.add_contours(nib.Nifti1Image((atlasData == 31).astype(int), affine=atlasAffine), filled=False, alpha=0.7, levels=[0.5], colors='b')

plotting.show()