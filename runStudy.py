import functionalNetworkConnectivity as f
import numpy as np
import cm
import utils
import matplotlib.pyplot as plt
import time
import os,sys

TR = 2
#f_lb = 0.02
#f_ub = 0.15
f_order = 2
measure = 'PC'
f_lb = 0.008
f_ub = 0.05
windowsSize = None
lagged = 3

#np.set_printoptions(precision=3)
#np.set_printoptions(threshold=np.inf)

namesNodes_node_to_node = ['Auditory', 'Cerebellum', 'DMN', 'ECN_L', 'ECN_R', 'Salience', 'Sensorimotor', 'Visual_L', 'Visual_M', 'Visual_O']
namesNodes_edge_to_edge = utils.toBuild_edgeNameList(namesNodes_node_to_node)

atlas = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz'
#atlas = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
#atlas = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/fconn_atlas_150_2mm.nii'
#atlas = '/home/jrudascas/Desktop/JoinROI2222.nii'
path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Control/'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Preproccesed/VS_UWS/'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Preproccesed/MCS/'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Controls/'
#path = '/home/jrudascas/Desktop/Projects/Dataset/Preproccesed/Temp'

FNC = f.functionalNetworkConnectivity()

#dynamic_correlation_matrix_atlas = FNC.dynamic_atlas(path=path, atlas=atlas, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, measure=measure, windowsSize=windowsSize)
#dynamic_edge_correlation_matrix = FNC.dynamic_build_edge_connectivity_matrix(dynamic_correlation_matrix_atlas)

#Static FNC
path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Control/'
group1 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, RSN=True)

windowsSize = 120

group2 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, RSN=True)
#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)

utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='manwhitneyu', outlier=-1.1)
utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='ttest', outlier=-1.1)

#G = cm.buildG_from_adjacency_matrix(np.mean(FNC.reduce_node_to_node_connectivity(static_correlation_matrix), axis=0))
#cm.draw_graph(G, node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
#edge_correlation_matrix = FNC.build_edge_connectivity_matrix(static_correlation_matrix)
#FNC.draw_correlation_matrix(np.mean(correlation_matrix3D2, axis=0), namesNodes_node_to_node, title="Correlation matrix - Node to Node", vmin=-1, vmax=1, returnPlot=True)
#FNC.draw_correlation_matrix(edge_correlation_matrix, namesNodes_edge_to_edge, title="Correlation matrix - Edge to Edge", vmin=-1, vmax=1, returnPlot=False)
#partition = cm.find_best_partition(adjacencyMatrix=edge_correlation_matrix)
#utils.toRelate_communities_to_nodes(partition, edge_correlation_matrix)


#Dynamic FNC
"""
dynamic_correlation_matrix = FNC.dynamic(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, measure=measure, windowsSize=windowsSize, RSN=True)
dynamic_edge_correlation_matrix = FNC.dynamic_build_edge_connectivity_matrix(dynamic_correlation_matrix)


hyperedgeSize = []
hyperedgeNumber = []
hyperedgeNodeDegree = []

size = int((1 + np.sqrt(1 + 8 * dynamic_edge_correlation_matrix[0].shape[0])) / 2)

hyperedgeNodeDegree = np.zeros((dynamic_edge_correlation_matrix.shape[0], size))

f1=open('Test.out', 'w+')

FNC.draw_correlation_matrix(np.mean(dynamic_edge_correlation_matrix, axis=0), title="Correlation matrix - Edge to Edge", vmin=-1, vmax=1, returnPlot=False)

for subject in range(dynamic_edge_correlation_matrix.shape[0]):
    dynamic_edge_correlation_matrix[subject, :, :] = abs(dynamic_edge_correlation_matrix[subject, :, :])

    partition = cm.find_best_partition(adjacencyMatrix=dynamic_edge_correlation_matrix[subject,:,:])

    c = utils.toRelate_communities_to_nodes(partition, dynamic_edge_correlation_matrix[subject,:,:], minimunOcurrence=1)

    # Hyperedge Number
    hyperedgeNumber.append(c.__len__())

    # Mean HyperedgeSize
    sum = 0
    for community, nodes in c.items():
        sum += len(c[community])

    hyperedgeSize.append(sum/c.__len__())

    listNode = range(size)
    for node in listNode:
        ocurrence = 0
        for community, nodes in c.items():
            if np.any(c[community] == node):
                ocurrence+= 1

        hyperedgeNodeDegree[subject, node] = ocurrence/c.__len__()

    print(hyperedgeNodeDegree[subject, :], file=f1)
    print('Fin subject # ' + str(subject))

print('6 ' + time.strftime("%H:%M:%S"))

print(hyperedgeSize, file=f1)
print(hyperedgeNumber, file=f1)

"""



# Revisar las comparaciones multiples
# Lagged
# Dynamic FNC

# Experimento 1 -> FNC - NBS
# Experimento 2 -> Poder de Clasificación Diagnostica
# Experimento 3 -> Zero Lagged
# Experimento 4 -> Autocorrelacion en distance correlation
# Experimento 5 -> Efecto de los filtros sobre los niveles de correlacion (Antes y Despues)