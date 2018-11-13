import numpy as np

import nibabel as nib
from nilearn import plotting
import utils as u
import os
import csv

path_t1_mni = '/home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T1_2mm_brain.nii.gz'
path_atlas_nmi = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
#path_atlas_nmi = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/2mm/AAL2.nii'
path_general = '/home/jrudascas/Desktop/Test/NewTest/'

crs_r_hc = path_general + 'crs_r_hc.csv'
crs_r_mcs = path_general + 'crs_r_mcs.csv'
crs_r_uws = path_general + 'crs_r_uws.csv'
crs_r = path_general + 'crs_r.csv'
name_td_map = 'TD_Map.nii'

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

crs_r_hc_values = np.array(crs_r_hc_values)
crs_r_mcs_values = np.array(crs_r_mcs_values)
crs_r_uws_values = np.array(crs_r_uws_values)

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
u.toFindStatisticDifference(np.asarray(list_group_td_map[0]), np.asarray(list_group_td_map[1]), threshold=0.0005)
print('HC vs UWS\n')
u.toFindStatisticDifference(np.asarray(list_group_td_map[0]), np.asarray(list_group_td_map[2]), threshold=0.0005)
print('MCS vs UWS\n')
u.toFindStatisticDifference(np.asarray(list_group_td_map[1]), np.asarray(list_group_td_map[2]), threshold=0.0005)

#display = plotting.plot_anat(os.path.join(pathInto, 'meanTDMap_' + group + '.nii'), cut_coords=[0,-60,45])
#dis.add_contours(nib.Nifti1Image((data_data_atlasatlas == 31).astype(int), affine=affine_atlas), filled=False, alpha=0.7, levels=[0.5], colors='b')

#plotting.show()

listRoiTo = range(121)
listRoiTo = [30,31]
import matplotlib.pyplot as plt

for roiTo in listRoiTo:
    g0 = np.asarray(list_group_td_map[0])[:, roiTo]
    g1 = np.asarray(list_group_td_map[1])[:, roiTo]
    g2 = np.asarray(list_group_td_map[2])[:, roiTo]

    rects1 = plt.bar(np.arange(3), height=[g0.mean(), g1.mean(), g2.mean()], width=0.5,
                     alpha=0.4,
                     color='b',
                     yerr=[g0.var(), g1.var(), g2.var()])

    plt.show()
    #u.plot_linear_regression((crs_r_hc_values, crs_r_mcs_values, crs_r_uws_values), (np.asarray(list_group_td_map[0])[:, roiTo], np.asarray(list_group_td_map[1])[:, roiTo],
    # np.asarray(list_group_td_map[2])[:, roiTo]), 'ROI: ' + str(roiTo))