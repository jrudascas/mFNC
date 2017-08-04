import numpy as np
from scipy import stats

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

def toFindStatisticDifference(x, y, outlier = None, measure='manwhitneyu', threshold = 0.05):

    print('\nDoing a multiple comparation by using ' + measure + ' test\n')

    if x.shape[-1] != y.shape[-1]:
        raise AttributeError('Shape incorrect')

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
        if p < therhold:
            print('Comparators ' + str(comparator + 1) + ' are statistically significant differences (' + str(p) + ')')

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
                namesNodes_edge_to_edge.append(namesNodes_node_to_node[i] + " - " + namesNodes_node_to_node[j] + " (" + str(cont) + ")")
                cont+=1

    return namesNodes_edge_to_edge

