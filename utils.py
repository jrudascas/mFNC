import numpy as np
from scipy import stats
from scipy import interpolate
import scipy.stats as sc
import distcorr as dc


def covariance(x, y):
    return np.dot(x,y)/x.shape[-1]

def mse(matrixA, matrixB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((matrixA.astype("float") - matrixB.astype("float")) ** 2)
    err /= float(matrixA.shape[0] * matrixA.shape[1] * np.mean(np.mean(matrixA)) * np.mean(np.mean(matrixB)))

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def to_interpolate_time_series(timeSerie, oldTR, newTR):
    x = np.round(np.linspace(0, len(timeSerie)*oldTR, len(timeSerie)))
    xToInterpolate = np.round(np.linspace(0, len(timeSerie)*oldTR, (len(timeSerie)+1)*oldTR/newTR))

    #interpolator = interpolate.BarycentricInterpolator(x, timeSerie)
    interpolator = interpolate.interp1d(x, timeSerie, kind='quadratic')
    return interpolator(xToInterpolate)

def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

def buildFeaturesVector(data):
    dimension = int((data.shape[1] * data.shape[1] - data.shape[1]) / 2)
    features = np.zeros((data.shape[0], dimension))
    cont = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if j > i:
                features[:, cont] = data[:, i, j]
                cont += 1
    return features

def to_build_features_vector_v2(data):
    dimension = data.shape[1] * data.shape[2]
    features = np.zeros((data.shape[0], dimension))
    cont = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
                features[:, cont] = data[:, i, j]
                cont += 1
    return features

def plot_linear_regression(tupla_x, tupla_y, title, save = False):
    from sklearn import linear_model
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    print('Plotting linear regression')

    regr = linear_model.LinearRegression()

    x_train = np.concatenate(tupla_x, axis=0)

    y_train = np.concatenate(tupla_y, axis=0)

    x_test = x_train

    regr.fit(np.transpose(np.matrix(x_train)), np.transpose(np.matrix(y_train)))

    y_pred = regr.predict(np.transpose(np.matrix(x_test)))

    plt.xlim(0, 30)
    #plt.ylim(-0.05, 0.05)

    plt.plot(x_test, y_pred, color='blue', linewidth=3)

    for element in range(len(tupla_x)):
        plt.scatter(tupla_x[element], tupla_y[element], alpha=0.5)

    plt.title(title + ' - ' + 'R2: ' + str(r2_score(y_train, y_pred)))
    if save:
        plt.savefig(title + '.png')
    plt.show()

def to_compute_time_series_similarity(time_serie_1, time_serie_2, measure):

    if measure not in ['PC', 'COV', 'DC']:
        raise AttributeError('Not is posible to estimate the measure ' + measure)

    if measure == 'COV':
        return covariance(time_serie_1, time_serie_2)
    elif measure == 'PC':
        return sc.pearsonr(time_serie_1, time_serie_2)[0]
    elif measure == 'DC':
        return dc.distcorr(time_serie_1, time_serie_2)


def to_extract_time_series(path_input, path_atlas = None, list_path_altas = None):
    import nibabel as nib

    img_fMRI = nib.load(path_input)
    data_fMRI = img_fMRI.get_data()

    time_series = []

    if path_atlas is not None:
        atlas_data = nib.load(path_atlas).get_data()
        index_roi = np.unique(atlas_data)
        for index in index_roi:
            if index != 0:
                time_series.append(np.mean(data_fMRI[atlas_data == index, :], axis=0))
    elif list_path_altas is not None:
        for path in list_path_altas:
            img_roi = nib.load(path)
            data_roi = img_roi.get_data()

            if data_roi.shape != data_fMRI.shape[:-1]:
                new_data = np.zeros((1, data_roi.shape[1], data_roi.shape[2]))
                data_roi = np.append(data_roi, new_data, axis=0)

            #time_series_filtered = []

            #for i in range(time_series_np.shape[0]):
            #    if not all(time_series_np[i,:] == 0):
            #        time_series_filtered.append(time_series_np[i,:])

            #time_series.append(np.mean(np.array(time_series_filtered), axis=0))

            time_series.append(np.mean(data_fMRI[data_roi != 0, :], axis=0))
    else:
        raise AttributeError('Both arguments, path_atlas and list_path_atlas, can not be None')

    return np.transpose(np.array(time_series))

def to_project_interval(x, e1, e2, s1, s2):
    return ((s1 - s2)/(e1 - e2))*(x-e1) + s1

def mean(data, outlier):
    average = np.zeros((data.shape[0], data.shape[1]))
    for index in range(data.shape[0]):
        for index2 in range(data.shape[1]):
            if all(data[index, index2, :] == outlier):
                average[index, index2] = outlier
            else:
                aux = np.concatenate((data[index, :index2, index2], data[index, index2+1:, index2]))

                average[index, index2] = np.mean(aux[aux != outlier])


    return average


def to_find_statistical_differences(x, y, outlier = None, measure='manwhitneyu', threshold = 0.05, to_correct = False, print_values = True):

    print('Doing a multiple comparation by using ' + measure + ' test')
    pLista = []
    if x.shape[-1] != y.shape[-1]:
        raise AttributeError('Shape incorrect')

    if to_correct:
        threshold = threshold/x.shape[-1]

    #if outlier is not None:

        #print('X')
        #print('')

        #for comparator in range(x.shape[-1]):
        #    if np.isnan(outlier):
        #        print(np.mean(x[~np.isnan(x[:, comparator]), comparator]))
                #print(x[~np.isnan(x[:, comparator]), comparator])
        #    else:
        #        print(np.mean(x[x[:, comparator] != outlier, comparator]))

        #print('Y')
        #print('')

        #for comparator in range(y.shape[-1]):
        #    if np.isnan(outlier):
        #        print(np.mean(y[~np.isnan(y[:, comparator]), comparator]))
                #print(y[~np.isnan(y[:, comparator]), comparator])
        #    else:
        #        print(np.mean(y[y[:, comparator] != outlier, comparator]))

    print('Number of comparator ' + str(x.shape[-1]))

    for comparator in range(x.shape[-1]):
        if measure == 'manwhitneyu':
            if outlier is None:

                if sum(x[:, comparator]) == 0 and sum(y[:, comparator]) == 0:
                    p = 1.
                else:
                    t, p = stats.mannwhitneyu(x[:, comparator], y[:, comparator])
            elif np.isnan(outlier):
                auxTemp1 = x[~np.isnan(x[:, comparator]), comparator]
                auxTemp2 = y[~np.isnan(y[:, comparator]), comparator]
                if np.all(auxTemp1 == auxTemp2[0]) or np.all(auxTemp2 == auxTemp1[0]):
                    p = 1.
                else:
                    t, p = stats.mannwhitneyu(x[~np.isnan(x[:, comparator]), comparator], y[~np.isnan(y[:, comparator]), comparator])
            else:
                t, p = stats.mannwhitneyu(x[x[:, comparator] != outlier, comparator], y[y[:, comparator] != outlier, comparator])
        elif measure == 'ttest':
            if outlier is None:
                t, p = stats.ttest_ind(x[:,comparator], y[:, comparator], equal_var=False)
            else:
                t, p = stats.ttest_ind(x[x[:, comparator] != outlier, comparator], y[y[:, comparator] != outlier, comparator], equal_var=False)

        #print("p: " + str(p))

        #print("p = " + str(p) + " Means: " + str(np.mean(x[x[:, comparator] != outlier, comparator])) + " - " + str(np.mean(y[y[:, comparator] != outlier, comparator])))
        pLista.append(p)
        if p < threshold:
            #print(x[~np.isnan(x[:, comparator]), comparator])
            #print(y[~np.isnan(y[:, comparator]), comparator])
            print('Comparator ' + str(comparator + 1) + ' (' + str(p) + ')')
            if print_values is True:
                print("x: " + str((x[~np.isnan(x[:, comparator]), comparator])))
                print("y: " + str((y[~np.isnan(y[:, comparator]), comparator])))
            #print(' - x:mean: ' + str(np.mean(x[~np.isnan(x[:, comparator]), comparator])) + ' x:std: ' + str(np.std(x[~np.isnan(x[:, comparator]), comparator])))
            #print(' - y:mean: ' + str(np.mean(y[~np.isnan(y[:, comparator]), comparator])) + ' y:std: ' + str(np.std(y[~np.isnan(y[:, comparator]), comparator])))
    return pLista

def toBuildMatrixDesign(pathIn, pathOut, maskEVs, maskThreadhold = None):
    import nibabel as nib
    fMRIdata = nib.load(pathIn).get_data()
    listEV = []
    for EV in maskEVs:
        if maskThreadhold is not None:
            mask = nib.load(EV).get_data() >= maskThreadhold

            if fMRIdata.shape[0:3] != mask.shape:
                newMask = np.insert(np.insert(np.insert(mask, 0, mask.shape[0] - 1, axis=0), 0, mask.shape[1] - 1, axis=1), 0, mask.shape[2] - 1, axis=2)
            else:
                newMask = mask

            listEV.append(np.mean(fMRIdata[newMask, :], axis=0))

    np.savetxt(pathOut + 'designMatrix.out', np.transpose(np.array(listEV)), fmt='%s')
    return pathOut + 'designMatrix.out'


def toFindStatisticDifference2(x1, x2, outlier = None, measure='manwhitneyu', threshold = 0.05):

    print('\nDoing a multiple comparation by using ' + measure + ' test\n')

    x = np.zeros((x1.shape[0], 1))
    y = np.zeros((x2.shape[0], 1))

    for i in range(x1.shape[0]):
        x[i] = np.sum(x1[i, (x1[i, :] != outlier)])/len((x1[i, (x1[i, :] != outlier)]))

    for i in range(x2.shape[0]):
        y[i] = np.sum(x2[i, (x2[i, :] != outlier)])/len((x2[i, (x2[i, :] != outlier)]))

    if x.shape[-1] != y.shape[-1]:
        raise AttributeError('Shape incorrect')

    therhold = threshold/x.shape[-1]

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    #print(np.mean(x))
    #print(np.mean(y))

    if measure == 'manwhitneyu':
        t, p = stats.mannwhitneyu(x, y)
    if measure == 'ttest':
        t, p = stats.ttest_ind(x, y, equal_var=False)

    print("p = " + str(p))

    if p < therhold:
       print('Are statistically significant differences (' + str(p) + ')')

def toFindStatisticDifference3(x1, x2, measure='manwhitneyu', threshold = 0.05):

    print('Finding by ' + measure)
    if measure == 'manwhitneyu':
        t, p = stats.mannwhitneyu(x1, x2)
    if measure == 'ttest':
        t, p = stats.ttest_ind(x1, x2, equal_var=False)

    print(p)
    if p < threshold:
       print('Are statistically significant differences (' + str(p) + ')')

def toFindStatisticDifference4(x, y, outlier = None, measure='manwhitneyu', threshold = 0.05):

    print('\nDoing a multiple comparation by using ' + measure + ' test\n')
    pLista = []

    #if x.shape[-1] != y.shape[-1]:
    #    raise AttributeError('Shape incorrect')

    therhold = threshold/x.shape[-1]

    for comparator in range(x.shape[-1]):
        if measure == 'manwhitneyu':
            if outlier is None:
                t, p = stats.mannwhitneyu(x[:,comparator], y[:, comparator])
            else:
                t, p = stats.mannwhitneyu(x[x[:, comparator] != outlier, comparator], y[y[:, comparator] != outlier, comparator])
        if measure == 'ttest':
            if outlier is None:
                t, p = stats.ttest_ind(x[:,comparator], y[:, comparator], equal_var=False)
            else:
                t, p = stats.ttest_ind(x[x[:, comparator] != outlier, comparator], y[y[:, comparator] != outlier, comparator], equal_var=False)
        print(p)
        pLista.append(p)
        if p < therhold:
            print('Comparators ' + str(comparator + 1) + ' are statistically significant differences (' + str(p) + ')')
    return pLista

def toRelate_communities_to_nodes(partition, edge_correlation_matrix, minimunOcurrence=2):
    hyperGraph = dict()

    for edge, community in partition.items():
        if community in hyperGraph:
            aux = list(hyperGraph.get(community))
            aux.append(find_nodes(edge, edge_correlation_matrix[0::].shape[-1])[0])
            aux.append(find_nodes(edge, edge_correlation_matrix[0::].shape[-1])[1])

            hyperGraph[community] = aux
        else:
            hyperGraph[community] = []
            aux = list(hyperGraph.get(community))
            aux.append(find_nodes(edge, edge_correlation_matrix[0::].shape[-1])[0])
            aux.append(find_nodes(edge, edge_correlation_matrix[0::].shape[-1])[1])

            hyperGraph[community] = aux

    nodeNumber = int((1 + np.sqrt(1 + 8 * edge_correlation_matrix.shape[0])) / 2)

    for community, nodes in hyperGraph.items():
        c = np.array(list(hyperGraph.get(community)))
        for node in range(nodeNumber):
            if len(np.where(c == node)[0]) < minimunOcurrence:
                c = c[np.where(c != node)]

        hyperGraph[community] = np.unique(c)
#        print(np.unique(c))
    return hyperGraph

def find_nodes(edgeIndex, numberEdge):
    nodeNumber = int((1 + np.sqrt(1+8*numberEdge))/2)
    count = 0
    for i in range(nodeNumber):
        for j in range(nodeNumber):
            if j > i:
                count += 1
            if count == edgeIndex:
                return i, j

def toBuild_edgeNameList(namesNodes_node_to_node):
    cont = 1
    namesNodes_edge_to_edge = []

    for i in range(len(namesNodes_node_to_node)):
        for j in range(len(namesNodes_node_to_node)):
            if j > i:
                namesNodes_edge_to_edge.append(namesNodes_node_to_node[i] + " - " + namesNodes_node_to_node[j])
                cont+=1

    return namesNodes_edge_to_edge

