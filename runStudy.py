import functionalNetworkConnectivity as f
import numpy as np
import cm
import utils
import matplotlib.pyplot as plt



TR = 2
#f_lb = 0.02
#f_ub = 0.15
f_order = 2
measure = 'PC'
f_lb = 0.008
f_ub = 0.05
windowsSize = 200

namesNodes_node_to_node = ['Auditory', 'Cerebellum', 'DMN', 'ECN_L', 'ECN_R', 'Salience', 'Sensorimotor', 'Visual_L', 'Visual_M', 'Visual_O']
namesNodes_edge_to_edge = utils.toBuild_edgeNameList(namesNodes_node_to_node)

atlas = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Control/'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
path = '/home/jrudascas/Desktop/Projects/Dataset/Preproccesed/'

FNC = f.functionalNetworkConnectivity()

dynamic_correlation_matrix_atlas = FNC.dynamic_atlas(path=path, atlas=atlas, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, measure=measure, windowsSize=windowsSize)
dynamic_edge_correlation_matrix = FNC.dynamic_build_edge_connectivity_matrix(dynamic_correlation_matrix_atlas)

#Static FNC
#static_correlation_matrix = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, measure=measure, RSN=True)
#G = cm.buildG_from_adjacency_matrix(np.mean(FNC.reduce_node_to_node_connectivity(static_correlation_matrix), axis=0))
#cm.draw_graph(G, node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
#edge_correlation_matrix = FNC.build_edge_connectivity_matrix(static_correlation_matrix)
#FNC.draw_correlation_matrix(np.mean(correlation_matrix3D2, axis=0), namesNodes_node_to_node, title="Correlation matrix - Node to Node", vmin=-1, vmax=1, returnPlot=True)
#FNC.draw_correlation_matrix(edge_correlation_matrix, namesNodes_edge_to_edge, title="Correlation matrix - Edge to Edge", vmin=-1, vmax=1, returnPlot=False)
#partition = cm.find_best_partition(adjacencyMatrix=edge_correlation_matrix)
#utils.toRelate_communities_to_nodes(partition, edge_correlation_matrix)

"""
#Dynamic FNC
dynamic_correlation_matrix = FNC.dynamic(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, measure=measure, windowsSize=windowsSize, RSN=True)
dynamic_edge_correlation_matrix = FNC.dynamic_build_edge_connectivity_matrix(dynamic_correlation_matrix)
"""

hyperedgeSize = []
hyperedgeNumber = []
hyperedgeNodeDegree = []

#size = int((1 + np.sqrt(1 + 8 * dynamic_edge_correlation_matrix[0].shape[0])) / 2)
size = int((dynamic_edge_correlation_matrix[0].shape[0]*dynamic_edge_correlation_matrix[0].shape[0] - dynamic_edge_correlation_matrix[0].shape[0])/2)

hyperedgeNodeDegree = np.zeros((dynamic_edge_correlation_matrix.shape[0], size))

for subject in range(dynamic_edge_correlation_matrix.shape[0]):
    dynamic_edge_correlation_matrix[subject, :, :] = abs(dynamic_edge_correlation_matrix[subject, :, :])

    #FNC.draw_correlation_matrix(dynamic_edge_correlation_matrix[num,:,:], namesNodes_edge_to_edge, title="Correlation matrix - Edge to Edge", vmin=-1, vmax=1, returnPlot=False)
    partition = cm.find_best_partition(adjacencyMatrix=dynamic_edge_correlation_matrix[subject,:,:])
    c = utils.toRelate_communities_to_nodes(partition, dynamic_edge_correlation_matrix[subject,:,:], minimunOcurrence=1)
    #c = partition
    # Hyperedge Number
    hyperedgeNumber.append(c.__len__())

    # Mean HyperedgeSize
    sum = 0
    for community, nodes in c.items():
        sum += len(c[community])

    hyperedgeSize.append(sum/c.__len__())

    #for node in range(size):
    for node in range(size):
        ocurrence = 0
        for community, nodes in c.items():
            if np.any(c[community] == node):
                ocurrence+= 1

        hyperedgeNodeDegree[subject, node] = ocurrence/c.__len__()
    print('Fin subject # ' + str(subject))

print('a')
print(np.mean(hyperedgeSize))
print(np.mean(hyperedgeNumber))
print(np.mean(hyperedgeNodeDegree, axis=0))


print('Fin')

# Revisar las comparaciones multiples
# Lagged
# Dynamic FNC

# Experimento 1 -> FNC - NBS
# Experimento 2 -> Poder de Clasificación Diagnostica
# Experimento 3 -> Zero Lagged
# Experimento 4 -> Autocorrelacion en distance correlation
# Experimento 5 -> Efecto de los filtros sobre los niveles de correlacion (Antes y Despues)