import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import utils as util
from scipy.signal import butter, filtfilt
from scipy import stats
import time
import distcorr as dc
import time
import numba

# Is necessary to install Tkinter -->sudo apt-get install python3-tk

class Core:
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
                connectivity_matrix[index1, index2] = sc.pearsonr(data[:, index1], data[:, index2])[
                    0] if measure == 'PC' else dc.distcorr(data[:, index1], data[:, index2])

        return connectivity_matrix

    def buildDynamicLaggedConnectivityMatrix(self, data, windowsSize=200, lagged=0, measure='PC'):

        #print('Building the correlation matrix and lagged map')

        timePoints, numberROI = data.shape

        if lagged == 0 or lagged is None:
            kCircular = []
            kCircular.append(0)
            temp2 = 0
        else:
            kCircular = range(-1 * lagged, lagged + 1, 1)
            temp2 = lagged

        indexROI = range(numberROI)

        if windowsSize is None:
            windowSlide = []
            windowSlide.append(0)
            rango3 = range(timePoints)
            temp1 = timePoints - 1
        else:
            windowSlide = range(0, timePoints - windowsSize, 1)
            rango3 = range(windowsSize)
            temp1 = windowsSize

        dynamicLaggedConnectivityMatrix = np.zeros((numberROI, numberROI, timePoints - temp1, 2 * temp2 + 1))
        timeDelayMatrix = np.zeros((numberROI, numberROI))
        amplitudeWeightedTimeDelayMatrix = np.zeros((numberROI, numberROI))

        listLaggeds = []

        for roi1 in indexROI:
            #print(str(float(roi1 / numberROI) * 10) + '%')
            for roi2 in indexROI:
                for slide in windowSlide:
                    indexWindows = np.array(rango3) + slide
                    timeSerie1 = data[indexWindows, roi1]
                    cont = 1
                    # plt.subplot(2 * lagged + 2, 1, cont)
                    # plt.plot(indexWindows, timeSerie1, label='timeSerie 1')

                    for lag in kCircular:
                        cont = cont + 1
                        timeSerie2 = np.roll(data[indexWindows, roi2], lag)

                        # plt.subplot(2*lagged+2,1, cont)
                        # plt.plot(indexWindows, timeSerie2, label='timeSerie 2')

                        if measure == 'PC':
                            dynamicLaggedConnectivityMatrix[roi1, roi2, slide, lag + lagged] = \
                                sc.pearsonr(timeSerie1, timeSerie2)[0]
                        elif measure == 'COV':
                            # dynamicLaggedConnectivityMatrix[roi1, roi2, slide, lag + lagged] = abs(np.cov(timeSerie1, timeSerie2, bias=True)[0][1])
                            dynamicLaggedConnectivityMatrix[roi1, roi2, slide, lag + lagged] = util.covariance(
                                timeSerie1, timeSerie2)
                        elif measure == 'DC':
                            dynamicLaggedConnectivityMatrix[roi1, roi2, slide, lag + lagged] = dc.distcorr(timeSerie1,
                                                                                                           timeSerie2)

                    # plt.show()

                # plt.plot(kCircular, abs(dynamicLaggedConnectivityMatrix[roi1, roi2, slide, :]), label='timeSerie 1')
                # plt.show()

                timeDelayMatrix[roi1, roi2] = np.where(
                    dynamicLaggedConnectivityMatrix[roi1, roi2, slide, :] == util.absmax(
                        dynamicLaggedConnectivityMatrix[roi1, roi2, slide, :]))[0][0] - lagged
                amplitudeWeightedTimeDelayMatrix[roi1, roi2] = util.absmax(
                    dynamicLaggedConnectivityMatrix[roi1, roi2, slide, :]) * (np.where(
                    dynamicLaggedConnectivityMatrix[roi1, roi2, slide, :] == util.absmax(
                        dynamicLaggedConnectivityMatrix[roi1, roi2, slide, :]))[0][0] - lagged)

                if (roi2 > roi1):
                    listLaggeds.append(np.where(dynamicLaggedConnectivityMatrix[roi1, roi2, slide, :] == util.absmax(
                        dynamicLaggedConnectivityMatrix[roi1, roi2, slide, :]))[0][0] - lagged)

        return dynamicLaggedConnectivityMatrix, listLaggeds, timeDelayMatrix, amplitudeWeightedTimeDelayMatrix

    @numba.jit
    def to_build_lagged_connectivity_matrix(self, data, lagged=0, measure='PC', tri_up = False):

        t = time.time()
        time_points, roit_number = data.shape

        if lagged == 0 or lagged is None:
            kCircular = [0]
            temp2 = 0
        else:
            kCircular = range(-1 * lagged, lagged + 1, 1)
            temp2 = lagged

        index_roi = range(roit_number)

        connectivity_matrix = np.zeros((roit_number, roit_number, 2 * temp2 + 1))
        td_matrix = np.zeros((roit_number, roit_number))
        awtd_matrix = np.zeros((roit_number, roit_number))

        for roi1 in index_roi:
            print(str(float(roi1 / roit_number) * 100) + '%')
            for roi2 in index_roi:
                if tri_up:
                    if roi2 > roi1:
                        for lag in kCircular:
                            connectivity_matrix[roi1, roi2, lag + lagged] = util.to_compute_time_series_similarity(
                                data[:, roi1], np.roll(data[:, roi2], lag), measure)

                        max_connection = util.absmax(connectivity_matrix[roi1, roi2, :])

                        td_matrix[roi1, roi2] = np.where(connectivity_matrix[roi1, roi2, :] == max_connection)[0][0] - lagged

                        awtd_matrix[roi1, roi2] = max_connection * td_matrix[roi1, roi2]
                else:
                    for lag in kCircular:
                        connectivity_matrix[roi1, roi2, lag + lagged] = util.to_compute_time_series_similarity(
                            data[:, roi1], np.roll(data[:, roi2], lag), measure)

                    max_connection = util.absmax(connectivity_matrix[roi1, roi2, :])

                    td_matrix[roi1, roi2] = np.where(connectivity_matrix[roi1, roi2, :] == max_connection)[0][0] - lagged

                    awtd_matrix[roi1, roi2] = max_connection * td_matrix[roi1, roi2]

        print("Time: " + str(time.time() - t))
        return util.absmax(connectivity_matrix, axis=-1), td_matrix, awtd_matrix

    def to_build_connectivity_matrix_2_groups(self, time_series_g1, time_series_g2, measure='PC'):

        connectivity_matrix = np.zeros((time_series_g1.shape[-1], time_series_g2.shape[-1]))

        index_roi_g1 = range(time_series_g1.shape[-1])
        index_roi_g2 = range(time_series_g2.shape[-1])

        for roi_g1 in index_roi_g1:
            aux_time_serie_g1 = time_series_g1[:, roi_g1]
            for roi_g2 in index_roi_g2:
                connectivity_matrix[roi_g1 - 1, roi_g2 - 1] = util.to_compute_time_series_similarity(aux_time_serie_g1,
                                                                                                     time_series_g2[:,
                                                                                                     roi_g2], measure)

        return connectivity_matrix

    def build_dynamic_connectivity_matrix(self, data, windowsSize=200, measure='PC'):
        shapeAxis2, shapeAxis = data.shape

        dynamic_connectivity_matrix = np.zeros((shapeAxis, shapeAxis, shapeAxis2 - windowsSize))

        for index1 in range(shapeAxis):
            for index2 in range(shapeAxis):
                for slide in range(0, shapeAxis2 - windowsSize, 1):
                    index = np.array(range(windowsSize)) + slide
                    dynamic_connectivity_matrix[index1, index2, slide] = \
                        sc.pearsonr(data[index, index1], data[index, index2])[0]
        # return dynamic_connectivity_matrix

        return dynamic_connectivity_matrix, np.max(dynamic_connectivity_matrix, axis=2)

    def reduce_node_to_node_connectivity(self, data, outlier=None, mandatory=True):
        recude_data = np.zeros((data.shape[1], data.shape[2]))

        for index1 in range(data.shape[1]):
            for index2 in range(data.shape[1]):

                if outlier is not None:
                    indexList = data[:, index1, index2] != outlier
                else:
                    indexList = data[:, index1, index2] != -np.pi

                if stats.ttest_1samp(data[indexList, index1, index2], 0.0)[1] < 0.05 or not mandatory:
                    recude_data[index1, index2] = np.mean(data[indexList, index1, index2])
                else:
                    recude_data[index1, index2] = 0

        return recude_data

    def reduce_neuronal_gof(self, data, neuronal, gof):
        recude_data = np.copy(data)
        for index1 in range(data.shape[1]):
            for index2 in range(data.shape[1]):
                if neuronal[index2] == 0 or neuronal[index1] == 0 or gof[index2] < 0.1 or gof[index1] < 0.1:
                    recude_data[index1, index2] = -1.1

                # np.where(data != -1.1)
        #                if (index2 > index1) and (stats.ttest_1samp(data[index1, index2], 0.0)[1] > 0.1):
        #                    recude_data[:, index1, index2] = 0.0

        return recude_data

    def estimate_variance_group(self, data):
        var = np.zeros((data.shape[-1], data.shape[-1]))
        for index1 in range(data.shape[1]):
            for index2 in range(data.shape[1]):
                if index2 > index1:
                    var[index1, index2] = np.var(data[:, index1, index2])
        return var

    def build_edge_connectivity_matrix(self, data):
        dimension = int((data.shape[1] * data.shape[1] - data.shape[1]) / 2)
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

                                    # if pearson[1] > (0.05/dimension2): #Bonferroni
                                    if pearson[1] > (
                                            1 / dimension2):  # false-positive correction for multiple comparisons,
                                        connectivity_matrix[cont1, cont2] = 0.0

                                cont2 += 1
                    cont1 += 1
        return connectivity_matrix

    # @jit
    def dynamic_build_edge_connectivity_matrix(self, data):
        print("\n" + "Started - Build Edge Connectivity Matrix based" + " --- " + time.strftime("%H:%M:%S") + "\n")

        dimension = int((data.shape[1] * data.shape[1] - data.shape[1]) / 2)
        dynamic_connectivity_matrix = np.zeros((data.shape[0], dimension, dimension))

        threshold = 0.05 / (int((dimension * dimension - dimension) / 2))
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
                                pearson = sc.pearsonr(data[subject, index1, index2, :],
                                                      data[subject, index3, index4, :])
                                if pearson[1] < threshold:
                                    dynamic_connectivity_matrix[subject, cont1, cont2] = pearson[0]
                                else:
                                    dynamic_connectivity_matrix[subject, cont1, cont2] = 0.0
                            cont2 += 1
                    cont1 += 1
        print("\n" + "Ended - Build Edge Connectivity Matrix based" + " --- " + time.strftime("%H:%M:%S") + "\n")
        return dynamic_connectivity_matrix

    def run2(self, time_series, tr, lag, new_tr=None, f_lb=0.005, f_ub=0.05, f_order=2, measure='PC', tri_up=False):

        for index in range(time_series.shape[1] - 1):
            time_series[:, index] = self.butter_bandpass_filter(time_series[:, index], f_lb, f_ub, tr, order=f_order)

        if new_tr is not None:
            # Interpolation in the time (to maximazate time scale)
            list_time_serie = list(np.transpose(time_series))
            new_time_series = [util.to_interpolate_time_series(time_serie, tr, new_tr) for time_serie in
                               list_time_serie]
            new_time_series = np.transpose(np.array(new_time_series))
            connectivity_matrix, td_matrix, awtd_matrix = self.to_build_lagged_connectivity_matrix(new_time_series,
                                                                                                   lagged=lag,
                                                                                                   measure=measure,
                                                                                                   tri_up=tri_up)
            return connectivity_matrix, td_matrix, awtd_matrix, new_tr
        else:
            connectivity_matrix, td_matrix, awtd_matrix = self.to_build_lagged_connectivity_matrix(time_series,
                                                                                                   lagged=lag,
                                                                                                   measure=measure,
                                                                                                   tri_up=tri_up)
            return connectivity_matrix, td_matrix, awtd_matrix, tr

    def run_2_groups(self, time_series_g1, time_series_g2, TR, f_lb=0.005, f_ub=0.05, f_order=2):

        for index in range(time_series_g1.shape[1] - 1):
            time_series_g1[:, index] = self.butter_bandpass_filter(time_series_g1[:, index], f_lb, f_ub, TR,
                                                                   order=f_order)

        for index in range(time_series_g2.shape[1] - 1):
            time_series_g2[:, index] = self.butter_bandpass_filter(time_series_g2[:, index], f_lb, f_ub, TR,
                                                                   order=f_order)

        # Interpolation in the time (to maximazate time scale)
        # newTR = 0.5
        # listTimeSerie = list(np.transpose(time_series_g1))
        # newTimeSeries = [util.toInterpolateTimeSerie(timeSerie, TR, newTR) for timeSerie in listTimeSerie]

        return self.to_build_connectivity_matrix_2_groups(time_series_g1, time_series_g2)

    def draw_correlation_matrix(self, correlation_matrix, title, namesTemplate=None, vmin=None, vmax=None,
                                returnPlot=False):
        plt.imshow(correlation_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(title)

        if namesTemplate is not None:
            plt.xticks(range(len(namesTemplate)), namesTemplate[0:], rotation=90)
            plt.yticks(range(len(namesTemplate)), namesTemplate[0:])

        if returnPlot is False:
            plt.show()
        else:
            return plt

    def run(self, path, TR, wSize, lag, f_lb=0.005, f_ub=0.05, f_order=2, measure='PC', reduce_neuronal=True,
            reductionMeasure='max', onlyRSN=True):

        correlation_matrix3D = []
        laggeds = []
        timeDelayMatrixs = []
        amplitudeWeightedTimeDelayMatrixs = []

        for dir in sorted(os.listdir(path)):
            print(dir)
            infile = open(os.path.join(os.path.join(path, dir), 'metadata.txt'), 'r')

            functionalImg = nib.load(
                os.path.join(os.path.join(os.path.join(path, dir), 'components'),
                             'icaAna_sub01_timecourses_ica_s1_.img'))
            functionalData = functionalImg.get_data()

            if onlyRSN is True:
                indexRSN = []
                neuronal = []
                gof = []
                for line in infile:
                    split = line.split(sep='\t')
                    indexRSN.append(int(split[1]) - 1)
                    neuronal.append(int(split[3]))
                    gof.append(float(split[2]))

                timeCourses = functionalData[:, indexRSN]
            else:
                timeCourses = functionalData

            for index in range(timeCourses.shape[1] - 1):
                timeCourses[:, index] = self.butter_bandpass_filter(timeCourses[:, index], f_lb, f_ub, TR,
                                                                    order=f_order)

            # Interpolation in the time (to maximazate time scale)
            newTR = 0.5
            listTimeSerie = list(np.transpose(timeCourses))
            newTimeSeries = [util.to_interpolate_time_series(timeSerie, TR, newTR) for timeSerie in listTimeSerie]
            newTimeSeries = np.transpose(timeCourses)

            correlation_matrix, listLaggeds, timeDelayMatrix, amplitudeWeightedTimeDelayMatrix = self.buildDynamicLaggedConnectivityMatrix(
                np.transpose(np.array(newTimeSeries)), windowsSize=wSize, lagged=lag, measure=measure)

            if (reduce_neuronal):
                if reductionMeasure == 'max':
                    correlation_matrix3D.append(
                        self.reduce_neuronal_gof(util.absmax(util.absmax(correlation_matrix, axis=-1), axis=-1),
                                                 neuronal,
                                                 gof))
                    timeDelayMatrixs.append(self.reduce_neuronal_gof(timeDelayMatrix * TR, neuronal, gof))
                elif reductionMeasure == 'mean':
                    correlation_matrix3D.append(
                        self.reduce_neuronal_gof(np.mean(np.mean(correlation_matrix, axis=-1), axis=-1), neuronal, gof))
                elif reductionMeasure == 'median':
                    correlation_matrix3D.append(
                        self.reduce_neuronal_gof(np.median(np.median(correlation_matrix, axis=-1), axis=-1), neuronal,
                                                 gof))
            else:
                if reductionMeasure == 'max':
                    correlation_matrix3D.append(util.absmax(util.absmax(correlation_matrix, axis=-1), axis=-1))
                elif reductionMeasure == 'mean':
                    correlation_matrix3D.append(np.mean(np.mean(correlation_matrix, axis=-1), axis=-1))
                elif reductionMeasure == 'median':
                    correlation_matrix3D.append(np.median(np.median(correlation_matrix, axis=-1), axis=-1))

            laggeds.append(listLaggeds)

            amplitudeWeightedTimeDelayMatrixs.append(amplitudeWeightedTimeDelayMatrix * TR)

        return np.array(correlation_matrix3D), np.array(laggeds), np.array(timeDelayMatrixs), np.array(
            amplitudeWeightedTimeDelayMatrixs)

    def dynamic(self, path, TR, f_lb, f_ub, windowsSize, f_order=2, measure='PC', RSN=True):
        print("dynamic FNC started\n")
        cont = 0

        correlation_matrix3D = []
        minTimeCourseSize = 9999

        for dir in sorted(os.listdir(path)):
            cont = cont + 1
            # print(os.path.join(os.path.join(path, dir), 'metadata.txt'))
            infile = open(os.path.join(os.path.join(path, dir), 'metadata.txt'), 'r')

            functional_file = nib.load(
                os.path.join(os.path.join(os.path.join(path, dir), 'components'),
                             'icaAna_sub01_timecourses_ica_s1_.img'))
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
                # plt.subplot(10,2, Index*2 + 1)
                # plt.plot(t, y, label='Filtered signal')
                # plt.subplot(10,2, Index*2 + 2)
                # plt.plot(t, TimeCourse[:,Index], label='Filtered signal')
                TimeCourse[:, Index] = y

            plt.show()

            correlation_matrix = \
                self.build_dynamic_connectivity_matrix(TimeCourse, windowsSize=windowsSize, measure=measure)[0]

            correlation_matrix3D.append(correlation_matrix)
            if correlation_matrix.shape[2] < minTimeCourseSize:
                minTimeCourseSize = correlation_matrix.shape[2]

        matrix = np.zeros(
            shape=[len(correlation_matrix3D), correlation_matrix3D[0].shape[0], correlation_matrix3D[0].shape[1],
                   minTimeCourseSize])

        print(minTimeCourseSize)

        for subjetc in range(len(correlation_matrix3D)):
            print(correlation_matrix3D[subjetc].shape)

            matrix[subjetc, :, :, :] = correlation_matrix3D[subjetc][:, :, range(minTimeCourseSize)]

        return matrix
        # print('Ending')

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
            # print("RR")

            # TimeCourse = self.usingROI(atlas, '/home/jrudascas/Desktop/AUX.nii')

            # print("TT")
            t = np.linspace(0, TimeCourse.shape[0], TimeCourse.shape[0], endpoint=False)

            # plt.subplot(911)
            # plt.plot(t, TimeCourse[:, 5], label='Filtered signal')
            # plt.subplot(912)
            # plt.plot(t, TimeCourse[:, 15], label='Filtered signal')
            # plt.subplot(913)
            # plt.plot(t, TimeCourse[:, 25], label='Filtered signal')
            # plt.subplot(914)
            # plt.plot(t, TimeCourse[:, 33], label='Filtered signal')

            for Index in range(TimeCourse.shape[1] - 1):
                TimeCourse[:, Index] = self.butter_bandpass_filter(TimeCourse[:, Index], f_lb, f_ub, TR, order=f_order)

            # plt.subplot(915)
            # plt.plot(t, TimeCourse[:, 5], label='Filtered signal')
            # plt.subplot(916)
            # plt.plot(t, TimeCourse[:, 15], label='Filtered signal')
            # plt.subplot(917)
            # plt.plot(t, TimeCourse[:, 25], label='Filtered signal')
            # plt.subplot(919)
            # plt.plot(t, TimeCourse[:, 33], label='Filtered signal')

            # plt.show()

            correlation_matrix = \
                self.build_dynamic_connectivity_matrix(TimeCourse, windowsSize=windowsSize, measure=measure)[0]

            correlation_matrix3D.append(correlation_matrix)

            if correlation_matrix.shape[2] < minTimeCourseSize:
                minTimeCourseSize = correlation_matrix.shape[2]

        matrix = np.zeros(
            shape=[len(correlation_matrix3D), correlation_matrix3D[0].shape[0], correlation_matrix3D[0].shape[1],
                   minTimeCourseSize])

        for subjetc in range(len(correlation_matrix3D)):
            matrix[subjetc, :, :, :] = correlation_matrix3D[subjetc][:, :, range(minTimeCourseSize)]

        print("Dynamic Atlas FNC started\n")
        return matrix

    # @jit
    # def usingROI(self, atlas, fmri):
    def usingROI(self, indexROI, timePoints, atlas_img, matrix):
        time_series = np.zeros((timePoints, len(indexROI)))
        for ROI in indexROI:
            time_series[:, ROI] = np.mean(matrix[:, (atlas_img == ROI)], axis=-1)

        # for volumen in range(matrix.shape[0]):
        #    print(np.mean(matrix[volumen, atlas_img == 13]))

        return time_series
