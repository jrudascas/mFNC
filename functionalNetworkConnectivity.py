import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import distcorr as dc
from scipy.signal import butter, filtfilt
from scipy import stats
import time
from numba import jit

#Is necessary to install Tkinter -->sudo apt-get install python3-tk

class functionalNetworkConnectivity:
    def __init__(self):
        pass

    def butter_bandpass(self, lowcut, highcut, fs, order=2):
        nyq = 1 / (2 * fs)
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def build_node_connectivity_matrix(self, data, measure='PC'):
        connectivity_matrix = np.zeros((data.shape[-1], data.shape[-1]))
        for index1 in range(data.shape[-1]):
            for index2 in range(data.shape[-1]):
                connectivity_matrix[index1, index2] = sc.pearsonr(data[:,index1], data[:,index2])[0] if measure == 'PC' else dc.distcorr(data[:,index1], data[:,index2])

        return connectivity_matrix

    def build_dynamic_lagged_connectivity_matrix(self, data, windowsSize=200, lagged = 0, measure='PC'):
        shapeAxis2, shapeAxis = data.shape

        if lagged == 0 or lagged is None:
            kCircular = []
            kCircular.append(0)
            temp2 = 1
        else:
            kCircular = range(-1*lagged, lagged+1, 1)
            temp2 = lagged

        rango1 = range(shapeAxis)

        if windowsSize is None:
            rango2 = []
            rango2.append(0)
            rango3 = range(shapeAxis2)
            temp1 = shapeAxis2 - 1
        else:
            rango2 = range(0, shapeAxis2 - windowsSize, 1)
            rango3 = range(windowsSize)
            temp1 = windowsSize

        dynamic_lagged_connectivity_matrix = np.zeros((shapeAxis, shapeAxis, shapeAxis2 - temp1, 2 * temp2 + 1))

        for index1 in rango1:
            for index2 in rango1:
                for slide in rango2:
                    cont = 0
                    for lag in kCircular:
                        index = np.array(rango3) + slide
                        t2 = np.roll(data[index, index2], lag)
                        dynamic_lagged_connectivity_matrix[index1, index2, slide, cont] = sc.pearsonr(data[index,index1], t2)[0]
                        cont+=1

        return dynamic_lagged_connectivity_matrix

    def build_dynamic_connectivity_matrix(self, data, windowsSize=200, measure='PC'):
        shapeAxis2, shapeAxis = data.shape

        dynamic_connectivity_matrix = np.zeros((shapeAxis, shapeAxis, shapeAxis2 - windowsSize))
        for index1 in range(shapeAxis):
            for index2 in range(shapeAxis):
                for slide in range(0, shapeAxis2 - windowsSize, 1):
                    index = np.array(range(windowsSize)) + slide
                    dynamic_connectivity_matrix[index1, index2, slide] = sc.pearsonr(data[index,index1], data[index,index2])[0]
        #return dynamic_connectivity_matrix

        return dynamic_connectivity_matrix, np.max(dynamic_connectivity_matrix, axis=2)

    def reduce_node_to_node_connectivity(self, data, outlier = None):
        recude_data = np.zeros((data.shape[1], data.shape[2]))

        for index1 in range(data.shape[1]):
            for index2 in range(data.shape[1]):

                if outlier is not None:
                    indexList = data[:,index1, index2] != outlier
                else:
                    indexList = data[:, index1, index2] != -np.pi

                if stats.ttest_1samp(data[indexList, index1, index2], 0.0)[1] < 0.05:
                    recude_data[index1, index2] = np.mean(data[indexList, index1, index2])

        return recude_data

    def reduce_neuronal_gof(self, data, neuronal, gof):
        recude_data = np.copy(data)
        for index1 in range(data.shape[1]):
            for index2 in range(data.shape[1]):
                if neuronal[index2] == 0 or neuronal[index1] == 0 or gof[index2] < 0.1 or gof[index1] < 0.1:
                    recude_data[index1, index2] = -1.1

                #np.where(data != -1.1)
#                if (index2 > index1) and (stats.ttest_1samp(data[index1, index2], 0.0)[1] > 0.1):
#                    recude_data[:, index1, index2] = 0.0

        return recude_data

    def estimate_variance_group(self, data):
        var = np.zeros((data.shape[-1], data.shape[-1]))
        for index1 in range(data.shape[1]):
            for index2 in range(data.shape[1]):
                if index2 > index1:
                   var[index1, index2] = np.var(data[:,index1, index2])
        return var

    def build_edge_connectivity_matrix(self, data):
        dimension = int((data.shape[1]*data.shape[1] - data.shape[1])/2)
        connectivity_matrix = np.zeros((dimension, dimension))
        cont1 = 0
        dimension2 = int((dimension * dimension - dimension) / 2)
        dimension2 = 1


        for index1 in range(data.shape[1]):
            for index2 in range(data.shape[1]):
                if (index2 > index1):
                    cont2 = 0
                    for index3 in range(data.shape[1]):
                        for index4 in range(data.shape[1]):
                            if index4 > index3:
                                if np.all(data[:, index1, index2]) and np.all(data[:, index3, index4]):
                                    pearson = sc.pearsonr(data[:, index1, index2], data[:, index3, index4])
                                    connectivity_matrix[cont1, cont2] = pearson[0]

                                    #if pearson[1] > (0.05/dimension2): #Bonferroni
                                    if pearson[1] > (1 / dimension2): #false-positive correction for multiple comparisons,
                                        connectivity_matrix[cont1, cont2] = 0.0

                                cont2+= 1
                    cont1+= 1
        return connectivity_matrix

    #@jit
    def dynamic_build_edge_connectivity_matrix(self, data):
        print("\n" + "Started - Build Edge Connectivity Matrix based" + " --- " + time.strftime("%H:%M:%S") + "\n")

        dimension = int((data.shape[1]*data.shape[1] - data.shape[1])/2)
        dynamic_connectivity_matrix = np.zeros((data.shape[0], dimension, dimension))

        threshold = 0.05/(int((dimension * dimension - dimension) / 2))
        rangeSize = range(data.shape[1])
        for subject in range(data.shape[0]):
            cont1 = 0
            print(subject)
            for index1 in rangeSize:
                for index2 in range(index1 + 1, data.shape[1]):
                    cont2 = 0
                    for index3 in rangeSize:
                        for index4 in range(index3 + 1, data.shape[1]):
                            if dynamic_connectivity_matrix[subject, cont2, cont1] == 0.0:
                                pearson = sc.pearsonr(data[subject, index1, index2, :], data[subject, index3, index4, :])
                                if pearson[1] < threshold:
                                    dynamic_connectivity_matrix[subject, cont1, cont2] = pearson[0]
                                else:
                                    dynamic_connectivity_matrix[subject, cont1, cont2] = 0.0
                            cont2+= 1
                    cont1+= 1
        print("\n" + "Ended - Build Edge Connectivity Matrix based" + " --- " + time.strftime("%H:%M:%S") + "\n")
        return dynamic_connectivity_matrix

    def draw_correlation_matrix(self, correlation_matrix, title, namesTemplate=None, vmin=None, vmax=None, returnPlot=False):

        plt.imshow(correlation_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(title)

        if namesTemplate is not None:
            plt.xticks(range(len(namesTemplate )), namesTemplate [0:], rotation=90)
            plt.yticks(range(len(namesTemplate )), namesTemplate [0:])

        if returnPlot is False:
            plt.show()
        else:
            return plt

    def run(self, path, TR, f_lb, f_ub, wSize, lag, f_order=5, measure='PC', RSN=True):
        print("\nFNC run started\n")
        cont = 0
        correlation_matrix3D = []
        minTimeCourseSize = 999999
        for dir in sorted(os.listdir(path)):
            cont = cont + 1
            print(dir)
            infile = open(os.path.join(os.path.join(path, dir), 'metadata.txt'), 'r')

            functional_file = nib.load(os.path.join(os.path.join(os.path.join(path, dir), 'components'), 'icaAna_sub01_timecourses_ica_s1_.img'))
            functional_img = functional_file.get_data()

            if RSN is True:
                RSNindex = []
                neuronal = []
                gof = []
                for line in infile:
                    split = line.split(sep='\t')
                    RSNindex.append(int(split[1]) - 1)
                    neuronal.append(int(split[3]))
                    gof.append(float(split[2]))

                TimeCourse = functional_img[:, RSNindex]
            else:
                TimeCourse = functional_img;

            t = np.linspace(0, TimeCourse.shape[0], TimeCourse.shape[0], endpoint=False)

            aux = np.copy(TimeCourse)

            for Index in range(TimeCourse.shape[1] - 1):
                y = self.butter_bandpass_filter(TimeCourse[:, Index], f_lb, f_ub, TR, order=f_order)
                #plt.subplot(10,2, RSNIndex*2 + 1)
                #plt.plot(t, y, label='Filtered signal')
                #plt.subplot(10,2, RSNIndex*2 + 2)
                #plt.plot(t, RSNTimeCourse[:,RSNIndex], label='Filtered signal')
                TimeCourse[:, Index] = y
            #plt.show()

            example = np.copy(aux[:, 0])
            size = example.shape[0]
            for a in aux[:, 0]:
                example[size - 1] = a
                size -= 1

            yyyj = self.butter_bandpass_filter(example, f_lb, f_ub, TR, order=f_order)

            correlation_matrix = self.build_dynamic_lagged_connectivity_matrix(TimeCourse, windowsSize=wSize, lagged=lag, measure=measure)

            #correlation_matrix3D.append(self.reduce_neuronal_gof(np.max(np.max(correlation_matrix, axis=-1), axis=-1), neuronal, gof))

            correlation_matrix3D.append(np.max(np.max(correlation_matrix, axis=-1), axis=-1))

        return np.array(correlation_matrix3D)

    def dynamic(self, path, TR, f_lb, f_ub, windowsSize, f_order=2, measure='PC', RSN=True):
        print("dynamic FNC started\n")
        cont = 0

        correlation_matrix3D = []
        minTimeCourseSize = 9999

        for dir in sorted(os.listdir(path)):
            cont = cont + 1
            #print(os.path.join(os.path.join(path, dir), 'metadata.txt'))
            infile = open(os.path.join(os.path.join(path, dir), 'metadata.txt'), 'r')

            functional_file = nib.load(os.path.join(os.path.join(os.path.join(path, dir), 'components'), 'icaAna_sub01_timecourses_ica_s1_.img'))
            functional_img = functional_file.get_data()

            if RSN is True:
                RSNindex = []
                for line in infile:
                    split = line.split(sep='\t')
                    RSNindex.append(int(split[1]) - 1)

                TimeCourse = functional_img[:, RSNindex]
            else:
                TimeCourse = functional_img;

            t = np.linspace(0, TimeCourse.shape[0], TimeCourse.shape[0], endpoint=False)

            for Index in range(TimeCourse.shape[1] - 1):
                y = self.butter_bandpass_filter(TimeCourse[:, Index], f_lb, f_ub, TR, order=f_order)
                #plt.subplot(10,2, Index*2 + 1)
                #plt.plot(t, y, label='Filtered signal')
                #plt.subplot(10,2, Index*2 + 2)
                #plt.plot(t, TimeCourse[:,Index], label='Filtered signal')
                TimeCourse[:, Index] = y

            plt.show()

            correlation_matrix = self.build_dynamic_connectivity_matrix(TimeCourse, windowsSize=windowsSize, measure=measure)[0]

            correlation_matrix3D.append(correlation_matrix)
            if correlation_matrix.shape[2] < minTimeCourseSize:
                minTimeCourseSize = correlation_matrix.shape[2]

        matrix = np.zeros(shape=[len(correlation_matrix3D), correlation_matrix3D[0].shape[0], correlation_matrix3D[0].shape[1], minTimeCourseSize])

        print(minTimeCourseSize)

        for subjetc in range(len(correlation_matrix3D)):
            print(correlation_matrix3D[subjetc].shape)

            matrix[subjetc,:,:,:] = correlation_matrix3D[subjetc][:,:,range(minTimeCourseSize)]

        return matrix
        #print('Ending')

    def dynamic_atlas(self, path, atlas, TR, f_lb, f_ub, windowsSize, f_order=5, measure='PC'):
        print("Dynamic Atlas FNC started\n")

        cont = 0

        correlation_matrix3D = []

        atlas_file = nib.load(atlas)
        atlas_img = atlas_file.get_data()

        indexROI = np.unique(atlas_img).astype(int)
        indexROI.sort(0)

        print("ROI Number = " + str(len(indexROI)))
        minTimeCourseSize = 9999

        listSubjects = sorted(os.listdir(path))
        for dir in listSubjects:
            print("Processing to " + dir + " folder" + " --- " + time.strftime("%H:%M:%S"))
            correlation_matrix4D = []
            filesT = os.walk(os.path.join(os.path.join(path, dir), 'data/rest/'))
            subPath = os.path.join(os.path.join(path, dir), 'data/rest/')

            cont = 0
            for root, dirs, files in filesT:
                listfiles = sorted(files)

                for image in listfiles:
                    if image.endswith('.img'):
                        correlation_matrix4D.append(nib.load(os.path.join(subPath, image)).get_data())

            matrix = np.array(correlation_matrix4D)
            timePoints = matrix.shape[0]

            TimeCourse = self.usingROI(indexROI, timePoints, atlas_img, matrix)
            #print("RR")

            #TimeCourse = self.usingROI(atlas, '/home/jrudascas/Desktop/AUX.nii')

            #print("TT")
            t = np.linspace(0, TimeCourse.shape[0], TimeCourse.shape[0], endpoint=False)

            #plt.subplot(911)
            #plt.plot(t, TimeCourse[:, 5], label='Filtered signal')
            #plt.subplot(912)
            #plt.plot(t, TimeCourse[:, 15], label='Filtered signal')
            #plt.subplot(913)
            #plt.plot(t, TimeCourse[:, 25], label='Filtered signal')
            #plt.subplot(914)
            #plt.plot(t, TimeCourse[:, 33], label='Filtered signal')

            for Index in range(TimeCourse.shape[1] - 1):
                TimeCourse[:, Index] = self.butter_bandpass_filter(TimeCourse[:, Index], f_lb, f_ub, TR, order=f_order)

            #plt.subplot(915)
            #plt.plot(t, TimeCourse[:, 5], label='Filtered signal')
            #plt.subplot(916)
            #plt.plot(t, TimeCourse[:, 15], label='Filtered signal')
            #plt.subplot(917)
            #plt.plot(t, TimeCourse[:, 25], label='Filtered signal')
            #plt.subplot(919)
            #plt.plot(t, TimeCourse[:, 33], label='Filtered signal')

            #plt.show()

            correlation_matrix = self.build_dynamic_connectivity_matrix(TimeCourse, windowsSize=windowsSize, measure=measure)[0]

            correlation_matrix3D.append(correlation_matrix)

            if correlation_matrix.shape[2] < minTimeCourseSize:
               minTimeCourseSize = correlation_matrix.shape[2]

        matrix = np.zeros(shape=[len(correlation_matrix3D), correlation_matrix3D[0].shape[0], correlation_matrix3D[0].shape[1], minTimeCourseSize])

        for subjetc in range(len(correlation_matrix3D)):
            matrix[subjetc,:,:,:] = correlation_matrix3D[subjetc][:,:,range(minTimeCourseSize)]

        print("Dynamic Atlas FNC started\n")
        return matrix

    #@jit
    #def usingROI(self, atlas, fmri):
    def usingROI(self, indexROI, timePoints, atlas_img, matrix):
        time_series = np.zeros((timePoints, len(indexROI)))
        for ROI in indexROI:
            time_series[:, ROI] = np.mean(matrix[:, (atlas_img == ROI)], axis=-1)

        #for volumen in range(matrix.shape[0]):
        #    print(np.mean(matrix[volumen, atlas_img == 13]))


        return time_series

    def test(self):
        from nilearn.connectome import ConnectivityMeasure
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]