import numpy as np

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

