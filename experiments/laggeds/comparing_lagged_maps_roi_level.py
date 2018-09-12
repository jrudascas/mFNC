import numpy as np

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn import plotting
from sklearn.metrics import mean_squared_error, r2_score
import utils as ut
import os
import warnings

path_t1_mni = '/home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T1_2mm_brain.nii.gz'
path_atlas_nmi = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'

path_general = '/home/jrudascas/Desktop/Test/'
name_td_map = 'TD_Map.nii'


img_atlas = nib.load(path_atlas_nmi)
data_atlas = img_atlas.get_data()
affine_atlas = img_atlas.get_affine()
list_group_td_map = []

index_roi = np.unique(data_atlas)

for group in sorted(os.listdir(path_general)):
    if os.path.isdir(os.path.join(path_general, group)):
        pathInto = os.path.join(path_general, group)
        TDMap = []
        meanTDMAPData = np.zeros(data_atlas.shape)

        cont = 0
        for dir in sorted(os.listdir(pathInto)):
            if os.path.isdir(os.path.join(os.path.join(path_general, group), dir)):
                print(os.path.join(os.path.join(path_general, group), dir))
                cont = cont + 1
                path_td_map = os.path.join(os.path.join(pathInto, dir), name_td_map)

                img_td_map = nib.load(path_td_map)
                data_td_map = img_td_map.get_data()
                affine_td_map = img_td_map.get_affine()

                roiLags = []

                for roi in index_roi:
                    roiLags.append(np.mean(data_td_map[data_atlas == roi]))
                TDMap.append(roiLags)

                meanTDMAPData = data_td_map + meanTDMAPData
        meanTDMAPData = meanTDMAPData/cont

        nib.save(nib.Nifti1Image(meanTDMAPData, affine=affine_td_map), os.path.join(pathInto, 'meanTDMap_' + group + '.nii'))
        dis = plotting.plot_stat_map(nib.Nifti1Image(meanTDMAPData, affine=affine_td_map), cut_coords=[0, -28, 3], bg_img=path_t1_mni,
                               vmax=0.5 * 4/2, output_file=os.path.join(pathInto, 'meanTDMap_' + group + '.png'))

        list_group_td_map.append(TDMap)

print('HC vs MCS\n')
ut.toFindStatisticDifference(np.asarray(list_group_td_map[0]), np.asarray(list_group_td_map[1]), threshold=0.01)
print('HC vs MCS\n')
ut.toFindStatisticDifference(np.asarray(list_group_td_map[0]), np.asarray(list_group_td_map[2]), threshold=0.01)
print('HC vs MCS\n')
ut.toFindStatisticDifference(np.asarray(list_group_td_map[1]), np.asarray(list_group_td_map[2]), threshold=0.01)

#display = plotting.plot_anat(os.path.join(pathInto, 'meanTDMap_' + group + '.nii'), cut_coords=[0,-60,45])
#dis.add_contours(nib.Nifti1Image((data_atlas == 31).astype(int), affine=affine_atlas), filled=False, alpha=0.7, levels=[0.5], colors='b')

#plotting.show()

regr = linear_model.LinearRegression()

listRoiTo = [2]
#listRoiTo = [2, 8, 11, 17, 18, 33, 39, 41]


for roiTo in listRoiTo:
    # Train the model using the training sets
    x_train = np.linspace(1, len(list_group_td_map[0]) + len(list_group_td_map[1]) + len(list_group_td_map[2]), len(list_group_td_map[0]) + len(list_group_td_map[1]) + len(list_group_td_map[2]))

    y_train = np.concatenate((np.asarray(list_group_td_map[0])[:, roiTo], np.asarray(list_group_td_map[1])[:, roiTo]), axis=0)
    y_train = np.concatenate((y_train, np.asarray(list_group_td_map[2])[:, roiTo]), axis=0)

    x_test = x_train

    regr.fit(np.transpose(np.matrix(x_train)), np.transpose(np.matrix(y_train)))

    y_pred = regr.predict(np.transpose(np.matrix(x_test)))

    plt.plot(x_test, y_pred, color='blue', linewidth=3)

    #print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
    #print('R2: %.2f' % r2_score(y_train, y_pred))

    plt.scatter(range(0, len(list_group_td_map[0])), np.asarray(list_group_td_map[0])[:, roiTo], alpha=0.5)
    plt.scatter(range(len(list_group_td_map[0]), len(list_group_td_map[0]) + len(list_group_td_map[1])), np.asarray(list_group_td_map[1])[:, roiTo], alpha=0.5)
    plt.scatter(range(len(list_group_td_map[0]) + len(list_group_td_map[1]), len(list_group_td_map[0]) + len(list_group_td_map[1]) + len(list_group_td_map[2])), np.asarray(list_group_td_map[2])[:, roiTo], alpha=0.5)
    plt.title('ROI: ' + str(roiTo) + ' - ' + 'R2: ' + str(r2_score(y_train, y_pred)))

    print(y_train)
    print(y_pred)

    plt.show()