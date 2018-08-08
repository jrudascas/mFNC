import os

def BET (file_in, file_out, parameters):
    command = 'bet ' + file_in + ' ' + file_out + ' ' + parameters
    #print(command)
    os.system(command)

def GLM (pathFileIn, pathDesignMatrix, pathOut, pathRes, pathMask):
    command = 'fsl_glm' + ' -i ' + pathFileIn + ' -d ' + pathDesignMatrix + ' -o ' + pathOut + ' --out_res=' + pathRes + ' -m ' +  pathMask
    #print(command)
    os.system(command)

def FAST (file_in, parameters):
    command = 'fast ' + parameters + ' ' + file_in
    os.system(command)

import nibabel as nib
import numpy as np
from dipy.align.reslice import reslice
import  utils as t
import os.path
from pathlib import Path

generalPath = '/home/runlab/data/COMA/'

pathMask = '/usr/share/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
dwiPath = 'data/functional/fmri.nii.gz'
t1Path = 'data/structural/fwmmprage.nii.gz'
maskPath = 'data/structural/fwc1mprage.nii.gz'
preResliced = 'data/_T1resliced.nii.gz'
preBET = 'data/_bet.nii.gz'
preDesignMatrix = 'data/'

timeDelayMapList = []
amplitudeWeightedTimeDelayMapList = []

for group in sorted(os.listdir(generalPath)):
    if os.path.isdir(os.path.join(generalPath, group)):
        pathInto = os.path.join(generalPath, group)
        for dir in sorted(os.listdir(pathInto)):
            pathfMRI = os.path.join(os.path.join(generalPath, dir), dwiPath)
            t1 = os.path.join(os.path.join(generalPath, dir), t1Path)
            greyMatter = os.path.join(os.path.join(generalPath, dir), maskPath)
            pathT1Resliced = os.path.join(os.path.join(generalPath, dir), preResliced)
            pathBET = os.path.join(os.path.join(generalPath, dir), preBET)
            pathDesignMatrix = os.path.join(os.path.join(generalPath, dir), preDesignMatrix)

            T1img = nib.load(t1)
            T1Data = T1img.get_data()
            T1Affine = T1img.affine

            print()
            print('-----------------------------------------')
            print(dir)
            print('-----------------------------------------')
            print()

            print("Reslicing")
            if not os.path.exists(pathT1Resliced):
                T1Resliced, affine2 = reslice(T1Data, T1Affine, T1img.header.get_zooms()[:3], (2., 2., 2.))
                mask_img = nib.Nifti1Image(T1Resliced.astype(np.float32), affine2)
                nib.save(mask_img, pathT1Resliced)

            print("BET")
            if not os.path.exists(pathBET):
                BET(pathT1Resliced, pathBET, '-m -f .4')

            print("FAST")
            if not os.path.exists(pathDesignMatrix + '_bet_pve_0.nii.gz'):
                FAST(pathBET, parameters='-n 3 -t 1')

            print('Building Matrix Design')
            if not os.path.exists(pathDesignMatrix + 'designMatrix.out'):
                maskEV = [pathDesignMatrix + '_bet_pve_0.nii.gz',
                          pathDesignMatrix + '_bet_pve_2.nii.gz']

                dm = t.toBuildMatrixDesign(pathfMRI, pathOut=pathDesignMatrix, maskEVs=maskEV, maskThreadhold=0.6)

            print('GLM')
            if not os.path.exists(pathDesignMatrix + 'functional/' + 'fmriGLM.nii.gz'):
                GLM(pathfMRI, dm, pathOut=pathDesignMatrix + 'ppp.txt', pathRes=pathDesignMatrix + 'functional/' + 'fmriGLM.nii.gz', pathMask=pathMask)