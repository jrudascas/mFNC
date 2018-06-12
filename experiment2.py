import functionalNetworkConnectivity as f
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
import dcor as dc
import scipy.stats as sc
import scipy.ndimage.morphology as mp
from nitime.viz import drawmatrix_channels
from scipy import stats
import utils as ut
import os

t1MNI = '/home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T1_2mm_brain.nii.gz'
atlas = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
indexROI = 30

path = '/media/jrudascas/ADATA HD720/DataSet/Belgium/Preprocessed/Test/'
TDMapName = 'TD_Map.nii'
maskPath = 'data/structural/fwc1mprage.nii.gz'

atlasImg = nib.load(atlas)
atlasData = atlasImg.get_data()

groupTDMap = []
for group in sorted(os.listdir(path)):
    print(group)
    pathInto = os.path.join(path, group)
    TDMap = []
    for dir in sorted(os.listdir(pathInto)):
        TDMapPath = os.path.join(os.path.join(pathInto, dir), TDMapName)

        TDMapImg = nib.load(TDMapPath)
        TDMapData = TDMapImg.get_data()
        TDMapffine = TDMapImg.get_affine()
        roiLags = []
        for roi in range(np.max(atlasData)):
            roiLags.append(np.mean(TDMapData[atlasData == roi]))
        TDMap.append(roiLags)
    groupTDMap.append(TDMap)

meanTDMAP = np.array(groupTDMap)

ut.toFindStatisticDifference(meanTDMAP[0,:,:], meanTDMAP[1,:,:], threshold=0.01)
ut.toFindStatisticDifference(meanTDMAP[0,:,:], meanTDMAP[2,:,:], threshold=0.01)
ut.toFindStatisticDifference(meanTDMAP[1,:,:], meanTDMAP[2,:,:], threshold=0.01)

print(meanTDMAP.shape)
x = np.arange(0.0, 20.0, 2.0)



