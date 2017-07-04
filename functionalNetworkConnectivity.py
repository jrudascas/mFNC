import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import distcorr as dc
import networkx as nx
import cm as cm
from scipy.signal import butter, lfilter
from scipy import stats

#Is necessary to install Tkinter -->sudo apt-get install python3-tk

class functionalNetworkConnectivity:
    def __init__(self):
        pass

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def build_node_connectivity_matrix(self, data, measure='PC'):
        connectivity_matrix = np.zeros((data.shape[-1], data.shape[-1]))
        for index1 in range(data.shape[-1]):
            for index2 in range(data.shape[-1]):
                connectivity_matrix[index1, index2] = sc.pearsonr(data[:,index1], data[:,index2])[0] if measure == 'PC' else dc.distcorr(data[:,index1], data[:,index2])

        return connectivity_matrix

    def build_dynamic_connectivity_matrix(self, data, windowsSize=200, measure='PC'):
        dynamic_connectivity_matrix = np.zeros((data.shape[-1], data.shape[-1], data.shape[0] - windowsSize))
        for index1 in range(data.shape[-1]):
            for index2 in range(data.shape[-1]):
                for slide in range(data.shape[0] - windowsSize):
                    index = np.array(range(windowsSize)) + slide
                    try:
                        dynamic_connectivity_matrix[index1, index2, slide] = sc.pearsonr(data[index,index1], data[index,index2])[0] if measure == 'PC' else dc.distcorr(data[:,index1], data[:,index2])
                    except Warning:
                        print(data[index,index1])
                        print(data[index, index2])

        return dynamic_connectivity_matrix, np.max(dynamic_connectivity_matrix, axis=2)

    def reduce_node_to_node_connectivity(self, data):
        recude_data = np.copy(data)
        for index1 in range(data.shape[1]):
            for index2 in range(data.shape[1]):
                if (index2 > index1) and (stats.ttest_1samp(data[:,index1, index2], 0.0)[1] > 0.1):
                    recude_data[:, index1, index2] = 0.0

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

        print(dimension2)

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

    def dynamic_build_edge_connectivity_matrix(self, data):
        dimension = int((data.shape[1]*data.shape[1] - data.shape[1])/2)
        dynamic_connectivity_matrix = np.zeros((data.shape[0], dimension, dimension))
        dimension2 = int((dimension * dimension - dimension) / 2)

        for subject in range(data.shape[0]):
            cont1 = 0
            for index1 in range(data.shape[1]):
                for index2 in range(data.shape[1]):
                    if (index2 > index1):
                        cont2 = 0
                        for index3 in range(data.shape[1]):
                            for index4 in range(data.shape[1]):
                                if index4 > index3:
                                    if np.all(data[subject, index1, index2, :]) and np.all(data[subject, index3, index4, :]):
                                        pearson = sc.pearsonr(data[subject, index1, index2, :], data[subject, index3, index4, :])
                                        dynamic_connectivity_matrix[subject, cont1, cont2] = pearson[0]

                                        if pearson[1] > (0.05/dimension2):
                                            dynamic_connectivity_matrix[subject, cont1, cont2] = 0.0

                                    cont2+= 1
                        cont1+= 1
        return dynamic_connectivity_matrix

    def draw_correlation_matrix(self, correlation_matrix, namesTemplate, title, vmin=None, vmax=None, returnPlot=False):

        plt.imshow(correlation_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(title)
        plt.xticks(range(len(namesTemplate )), namesTemplate [0:], rotation=90)
        plt.yticks(range(len(namesTemplate )), namesTemplate [0:])

        if returnPlot is False:
            plt.show()
        else:
            return plt

    def run(self, path, TR, f_lb, f_ub, f_order=5, measure='PC', RSN=True):
        print("FNC run started\n")
        cont = 0
        correlation_matrix3D = []

        for dir in sorted(os.listdir(path)):
            cont = cont + 1
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
                #plt.subplot(10,2, RSNIndex*2 + 1)
                #plt.plot(t, y, label='Filtered signal')
                #plt.subplot(10,2, RSNIndex*2 + 2)
                #plt.plot(t, RSNTimeCourse[:,RSNIndex], label='Filtered signal')
                TimeCourse[:, Index] = y
            #plt.show()

            correlation_matrix = self.build_node_connectivity_matrix(TimeCourse, measure=measure)

            correlation_matrix3D.append(correlation_matrix)

        return np.array(correlation_matrix3D)
        #print('Ending')


    def dynamic(self, path, TR, f_lb, f_ub, windowsSize, TCSize = 297, f_order=5, measure='PC', RSN=True):
        print("dynamic FNCstarted\n")
        cont = 0

        correlation_matrix3D = []

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
                #plt.subplot(10,2, RSNIndex*2 + 1)
                #plt.plot(t, y, label='Filtered signal')
                #plt.subplot(10,2, RSNIndex*2 + 2)
                #plt.plot(t, RSNTimeCourse[:,RSNIndex], label='Filtered signal')
                TimeCourse[:, Index] = y
            #plt.show()

            correlation_matrix = self.build_dynamic_connectivity_matrix(TimeCourse, windowsSize=windowsSize, measure=measure)[0]
            minTimeCourseSize = 9999
            correlation_matrix3D.append(correlation_matrix)
            if correlation_matrix.shape[2] < minTimeCourseSize:
                minTimeCourseSize = correlation_matrix.shape[2]

        matrix = np.zeros(shape=[len(correlation_matrix3D), correlation_matrix3D[0].shape[0], correlation_matrix3D[0].shape[1], minTimeCourseSize])

        for subjetc in range(len(correlation_matrix3D)):
            matrix[subjetc,:,:,:] = correlation_matrix3D[subjetc][:,:,range(minTimeCourseSize)]

        return matrix
        #print('Ending')


    def dynamic_atlas(self, path, atlas, TR, f_lb, f_ub, windowsSize, f_order=5, measure='PC'):
        print("Dynamic Atlas FNC started\n")

        cont = 0

        correlation_matrix3D = []
        correlation_matrix4D = []

        atlas_file = nib.load(atlas)
        atlas_img = atlas_file.get_data()

        indexROI = np.unique(atlas_img)
        indexROI.sort(0)
        print("ROI Number = " + str(len(indexROI)))

        for dir in sorted(os.listdir(path)):
            cont = cont + 1
            print("Processing to " + dir + " folder")

            filesT = os.walk(os.path.join(os.path.join(path, dir), 'data/rest/'))
            for root, dirs, files in filesT:
                for image in sorted(files):
                    if image.endswith('.img'):
                        functional_image = nib.load(os.path.join(os.path.join(os.path.join(path, dir), 'data/rest/'), image))
                        functional_data = functional_image.get_data()
                        correlation_matrix4D.append(functional_data)
                    pass
                pass

            matrix = np.array(correlation_matrix4D)
            TimeCourse = np.zeros((matrix.shape[0], len(indexROI)))

            for ROI in indexROI:
                if ROI != 0:
                    ROIimg = (atlas_img == ROI)
                    for volume in range(matrix.shape[0]):
                        TimeCourse[volume, ROI] = np.mean(matrix[volume, ROIimg])

            for Index in range(TimeCourse.shape[1] - 1):
                TimeCourse[:, Index] = self.butter_bandpass_filter(TimeCourse[:, Index], f_lb, f_ub, TR, order=f_order)

            correlation_matrix = self.build_dynamic_connectivity_matrix(TimeCourse, windowsSize=windowsSize, measure=measure)[0]

            correlation_matrix3D.append(correlation_matrix)
            minTimeCourseSize = 9999
            if correlation_matrix.shape[2] < minTimeCourseSize:
               minTimeCourseSize = correlation_matrix.shape[2]

        matrix = np.zeros(shape=[len(correlation_matrix3D), correlation_matrix3D[0].shape[0], correlation_matrix3D[0].shape[1], minTimeCourseSize])

        for subjetc in range(len(correlation_matrix3D)):
            matrix[subjetc,:,:,:] = correlation_matrix3D[subjetc][:,:,range(minTimeCourseSize)]

        return matrix