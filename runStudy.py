import functionalNetworkConnectivity as f
import utils
import sys
import plotGallery as pg
import numpy as np
from nilearn import plotting as plott
import matplotlib.pyplot as plt

TR = 2
f_order = 2
outlier = -1.1
f_lb = 0.008
f_ub = 0.05
umbral = 0.6

#np.set_printoptions(precision=3)
#np.set_printoptions(threshold=np.inf)

namesNodes_node_to_node = ['Auditory', 'Cerebellum', 'DMN', 'ECL', 'ECR', 'Salience', 'SensoriMotor', 'Vis_Lateral', 'Vis_Medial', 'Vis_Occipital']
#namesNodes_node_to_node = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '1O']
namesNodes_edge_to_edge = utils.toBuild_edgeNameList(namesNodes_node_to_node)

coords = [[-60, 0, 20],
          [0, -60, -50],
          [0, 50, 10],
          [-60, -50, 50],
          [60, -50, 50],
          [0, 20, 60],
          [0, -20, 70],
          [50, -80, -10],
          [0, -80, 0],
          [0, -100, 10]
          ]

#########################################################################################################################
"""

#Max vs Mean lagged  -> Suplementary Material

print("Experiment 1\n")

FNC = f.functionalNetworkConnectivity()
lagged = 3
windowsSize = None
measure = 'PC'
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Control/'
group1 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)
group2 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'mean', RSN=True)

#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='manwhitneyu', outlier=outlier)
pg.barchart(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), title= "Max Static Lagged correlation vs Mean Static Lagged correlation", labelGroup1="Max Static Lagged", labelGroup2="Mean Static Lagged", xLabel="# Connections", yLabel="Connectivity Mean", outlier=outlier)
"""
#########################################################################################################################

#########################################################################################################################

#Static Lagged vs Static NonLagged
# Window size = 240

print("Experiment 2\n")

p1 = []
p2 = []
p3 = []
p4 = []

for slide in range(300, 330, 30):
    print()
    slide = 280

    FNC = f.functionalNetworkConnectivity()
    lagged = 3
    measure = 'PC'
    windowsSize = slide

    path = '/home/runlab/data/COMA/COMA_ICA/MCS/'
    group1, laggeds1, TD1, AWTD1 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = False, reductionMeasure = 'max', onlyRSN=True)

    path = '/home/runlab/data/COMA/COMA_ICA/UWS/'
    group2, laggeds2, TD2, AWTD2 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = False, reductionMeasure = 'max', onlyRSN=True)

    path = '/home/runlab/data/COMA/COMA_ICA/Control/'
    group3, laggeds3, TD3, AWTD3 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = False, reductionMeasure = 'max', onlyRSN=True)

    pg.fivethirtyeightPlot(laggeds1, laggeds2, group3=laggeds3, lag=lagged, labelFeautures=namesNodes_edge_to_edge,
                           save='ThreadsLagPC.png')

    new = np.zeros((group1.shape[0], group1.shape[1]))
    new2 = np.zeros((group2.shape[0], group2.shape[1]))

    new3 = np.zeros((group1.shape[0], group1.shape[1]))
    new4 = np.zeros((group2.shape[0], group2.shape[1]))

    new5 = np.zeros((group3.shape[0], group3.shape[1]))
    new6 = np.zeros((group3.shape[0], group3.shape[1]))

    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            if np.alltrue(group1[i, j, :] == outlier):
                new[i, j] = np.nan
                new3[i, j] = np.nan
            else:
                aux = group1[i, j, group1[i, j, :] != outlier]
                if np.alltrue(aux == 1.):
                    new[i, j] = np.nan
                    new3[i, j] = np.nan
                else:
                    new[i, j] = np.mean(aux[aux != 1.])
                    auxTemp1 = aux[aux != 1.]
                    new3[i,j] = len(np.where(abs(auxTemp1) > umbral)[0])
                #new[i, j] = np.mean(aux)

    for i in range(new2.shape[0]):
        for j in range(new2.shape[1]):
            if np.alltrue(group2[i, j, :] == outlier):
                new2[i, j] = np.nan
                new4[i, j] = np.nan
            else:
                aux2 = group2[i, j, group2[i, j, :] != outlier]
                if np.alltrue(aux2 == 1.):
                    new2[i, j] = np.nan
                    new4[i, j] = np.nan
                else:
                    new2[i, j] = np.mean(aux2[aux2 != 1.])
                    auxTemp2 = aux2[aux2 != 1.]
                    new4[i,j] = len(np.where(abs(auxTemp2) > umbral)[0])

    for i in range(new5.shape[0]):
        for j in range(new6.shape[1]):
            if np.alltrue(group3[i, j, :] == outlier):
                new5[i, j] = np.nan
                new6[i, j] = np.nan
            else:
                aux3 = group3[i, j, group3[i, j, :] != outlier]
                if np.alltrue(aux3 == 1.):
                    new5[i, j] = np.nan
                    new6[i, j] = np.nan
                else:
                    new5[i, j] = np.mean(aux3[aux3 != 1.])
                    auxTemp3 = aux3[aux3 != 1.]
                    new6[i, j] = len(np.where(abs(auxTemp3) > umbral)[0])

    #cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size=2000, alpha=0.5, scale=20, namesNodes=namesNodes_node_to_node, save="MCS1")
    #cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group2, outlier=-1.1), allNode=True), node_size=2000, alpha=0.5, scale=20, namesNodes=namesNodes_node_to_node, save="VS1")
    # pg.barchart(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), title= "Non-Lagged Linear Correlation", labelGroup1="MCS", labelGroup2="VS/UWS", xLabel="", yLabel="Average Connectivity Level", outlier=outlier, save="fig11.png", labelFeautures=namesNodes_edge_to_edge)

    print("Test Graph MCS UWS\n")
    pList1 = utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='manwhitneyu', outlier=outlier)

    print("Test Graph HC MCS\n")
    pList1 = utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group3),
                                             measure='manwhitneyu', outlier=outlier)

    print("Test Graph HC UWS\n")
    pList1 = utils.toFindStatisticDifference(utils.buildFeaturesVector(group2), utils.buildFeaturesVector(group3),
                                             measure='manwhitneyu', outlier=outlier)

    """
    print("Test Conectividad Promedio\n")
    pList1 = utils.toFindStatisticDifference(new, new2, measure='manwhitneyu', outlier=np.nan)

    print('--------------------')
    pList1 = utils.toFindStatisticDifference(new, new5, measure='manwhitneyu', outlier=np.nan)

    print("Test Hiperconectividades\n")
    pList2 = utils.toFindStatisticDifference(new3, new4, measure='manwhitneyu', outlier=np.nan)

    print ('--------------------')
    pList2 = utils.toFindStatisticDifference(new3, new6, measure='manwhitneyu', outlier=np.nan)
    """

    print("Lista de P")
    print(pList1)

    correlation_matrix1 = FNC.reduce_node_to_node_connectivity(group1, outlier=outlier, mandatory = True)
    correlation_matrix2 = FNC.reduce_node_to_node_connectivity(group2, outlier=outlier, mandatory = True)
    correlation_matrix3 = FNC.reduce_node_to_node_connectivity(group3, outlier=outlier, mandatory=True)

    np.fill_diagonal(correlation_matrix1, 0)
    np.fill_diagonal(correlation_matrix2, 0)
    np.fill_diagonal(correlation_matrix3, 0)



    fig, ax = plt.subplots()

    for q in range(7):
        dataAux1 = group1[q, :, :].copy()
        dataAux2 = group2[q, :, :].copy()
        dataAux3 = group3[q, :, :].copy()

        dataAux1[dataAux1 == outlier] = 0
        dataAux2[dataAux2 == outlier] = 0
        dataAux3[dataAux3 == outlier] = 0

        iu1 = np.triu_indices(10, k=1)

        if np.sum(dataAux1[iu1]) != 0.0:

            new = np.zeros((lagged * 2 + 1, 1))
            for i in range(new.shape[0]):
                new[i, 0] = np.count_nonzero(laggeds1[q, :] == i - lagged)
            new = new/45

            plt.bar(range(7), height=new, alpha=0.5, color='b')
            plt.title('MCS ' + str(q))
            plt.show()

            """
            plott.plot_connectome(dataAux1,
                                  coords,
                                  node_color=['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                                  node_size=200, edge_vmax=.8, title='MCS ' + str(q) , edge_vmin=-0.8, colorbar=True)
            """

        if np.sum(dataAux2[iu1]) != 0.0:
            new = np.zeros((lagged * 2 + 1, 1))
            for i in range(new.shape[0]):
                new[i, 0] = np.count_nonzero(laggeds2[q, :] == i - lagged)
            new = new/45

            plt.bar(range(7), height=new, alpha=0.5, color='b')
            plt.title('UWS ' + str(q))
            plt.show()

            """
            plott.plot_connectome(dataAux2,
                                  coords,
                                  node_color=['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                                  node_size=200, edge_vmax=.8, title='UWS ' + str(q), edge_vmin=-0.8, colorbar=True)
            """

        if np.sum(dataAux3[iu1]) != 0.0:

            new = np.zeros((lagged * 2 + 1, 1))
            for i in range(new.shape[0]):
                new[i, 0] = np.count_nonzero(laggeds3[q, :] == i - lagged)
            new = new/45

            plt.bar(range(7), height=new, alpha=0.5, color='b')
            plt.title('HC ' + str(q))
            plt.show()

            """
            plott.plot_connectome(dataAux3,
                                  coords,
                                  node_color=['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                                  node_size=200, edge_vmax=.8, title='HC ' + str(q), edge_vmin=-0.8, colorbar=True)
            """

    plott.plot_connectome(correlation_matrix1,
                          coords,
                          node_color = ['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                          node_size=200, edge_vmax=.8, title='MCS', edge_vmin=-0.8, colorbar = True)

    plott.plot_connectome(correlation_matrix2,
                          coords,
                          node_color = ['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                          node_size=200, edge_vmax=.8, title='UWS', edge_vmin=-0.8, colorbar = True)

    plott.plot_connectome(correlation_matrix3,
                          coords,
                          node_color = ['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                          node_size=200, edge_vmax=.8, title='HC', edge_vmin=-0.8, colorbar = True)

    #plotting.plot_connectome(correlation_matrix1,
    #                         coords, edge_threshold='90%',
    #                         title=title,
    #                         edge_vmax=.9, edge_vmin=-.9)

    plott.plot_matrix(correlation_matrix1, labels=namesNodes_node_to_node, vmax=0.8, vmin=-0.8)
    plott.plot_matrix(correlation_matrix2, labels=namesNodes_node_to_node, vmax=0.8, vmin=-0.8)
    plott.plot_matrix(correlation_matrix3, labels=namesNodes_node_to_node, vmax=0.8, vmin=-0.8)
    plott.show()

    sys.exit(0)

    print()
    print("Non-Linear")
    print()
    FNC = f.functionalNetworkConnectivity()
    lagged = 3
    measure = 'DC'
    windowsSize = slide
    f_lb = 0.008
    f_ub = 0.05

    path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
    group3, laggeds3 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

    path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
    group4, laggeds4 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)


    new = np.zeros((group3.shape[0], group3.shape[1]))
    new2 = np.zeros((group4.shape[0], group4.shape[1]))
    new3 = np.zeros((group3.shape[0], group3.shape[1]))
    new4 = np.zeros((group4.shape[0], group4.shape[1]))

    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            if np.alltrue(group3[i, j, :] == outlier):
                new[i, j] = np.nan
                new3[i, j] = np.nan
            else:
                aux = group3[i, j, group3[i, j, :] != outlier]
                if np.alltrue(aux == 1.):
                    new[i, j] = np.nan
                    new3[i, j] = np.nan
                else:
                    new[i, j] = np.mean(aux[aux != 1.])
                    auxTemp = aux[aux != 1.]
                    new3[i,j] = len(np.where(abs(auxTemp) > umbral)[0])
                #new[i, j] = np.mean(aux)

    for i in range(new2.shape[0]):
        for j in range(new2.shape[1]):
            if np.alltrue(group4[i, j, :] == outlier):
                new2[i, j] = np.nan
                new4[i, j] = np.nan
            else:
                aux2 = group4[i, j, group4[i, j, :] != outlier]
                if np.alltrue(aux2 == 1.):
                    new2[i, j] = np.nan
                    new4[i, j] = np.nan
                else:
                    new2[i, j] = np.mean(aux2[aux2 != 1.])
                    auxTemp2 = aux2[aux2 != 1.]
                    new4[i,j] = len(np.where(abs(auxTemp2) > umbral)[0])
                #new2[i, j] = np.mean(aux2)

    print("Test Conectividad Promedio")
    pList3 = utils.toFindStatisticDifference(new, new2, measure='manwhitneyu', outlier=np.nan)

    print("Test Hiperconectividades")
    pList4 = utils.toFindStatisticDifference(new3, new4, measure='manwhitneyu', outlier=np.nan)

#    p1.append(np.mean(pList1))
#    p2.append(np.mean(pList2))
#    p3.append(np.mean(pList3))
#    p4.append(np.mean(pList4))

print("Listas de p")

print(p1)
print(p2)
print(p3)
print(p4)
#pList2 = utils.toFindStatisticDifference(utils.buildFeaturesVector(group3), utils.buildFeaturesVector(group4), measure='manwhitneyu', outlier=outlier)

#utils.toFindStatisticDifference3(pList1, pList2, measure='manwhitneyu', threshold = 0.05)

#pg.fivethirtyeightPlot(laggeds3, laggeds4, lag=lagged, save='ThreadsLagDC.png')

#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group2, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group3, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group4, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)

#pg.barchart(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), title= "Non-Lagged Linear Correlation", labelGroup1="MCS", labelGroup2="VS/UWS", xLabel="", yLabel="Average Connectivity Level", outlier=outlier, save="fig3.png", labelFeautures=namesNodes_edge_to_edge)
#pg.barchart(utils.buildFeaturesVector(group3), utils.buildFeaturesVector(group4), title= "Lagged Non-Linear Correlation", labelGroup1="MCS", labelGroup2="VS/UWS", xLabel="", yLabel="Average Connectivity Level", outlier=outlier, save="fig4.png", labelFeautures=namesNodes_edge_to_edge)

print("TTT")
pg.barchart4(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "Non-Lagged Linear Correlation vs Lagged Linear Correlation", labelGroup1="Non-Lagged Linear - MCS", labelGroup2="Non-Lagged Linear - VS/UWS", labelGroup3="Lagged Linear - MCS", labelGroup4="Lagged Linear - VS/UWS", xLabel="", yLabel="Average Connectivity Values", outlier=outlier, save="fig2.png")
#########################################################################################################################

#########################################################################################################################
"""
#Static vs Dynamic

print("Experiment 3\n")

FNC = f.functionalNetworkConnectivity()
lagged = 0
measure = 'DC'
windowsSize = None
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group1 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group2 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='manwhitneyu', outlier=outlier)

FNC = f.functionalNetworkConnectivity()
lagged = 0
measure = 'DC'
windowsSize = 120
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group3 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group4 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
utils.toFindStatisticDifference(utils.buildFeaturesVector(group3), utils.buildFeaturesVector(group4), measure='manwhitneyu', outlier=outlier)

pg.barchart2(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonLagged Static correlation vs NonLagged Dynamic correlation", labelGroup1="NLSC MCS", labelGroup2="NLSC VS/UWS", labelGroup3="NLDC MCS", labelGroup4="NLDC VS/UWS", xLabel="# Connections", yLabel="Mean Connectivity", outlier=outlier)
pg.barchart3(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonLagged Static correlation vs NonLagged Dynamic correlation", labelGroup1="NLSC Mean MCS", labelGroup2="NLSC Mean VS/UWS", labelGroup3="NLDC Mean MCS", labelGroup4="NLDC Mean VS/UWS", labelGroup5="NLSC Std MCS", labelGroup6="NLSC Std VS/UWS", labelGroup7="NLDC Std MCS", labelGroup8="NLDC Std VS/UWS", xLabel="", yLabel="Mean Values", outlier=outlier)
"""
#########################################################################################################################


#########################################################################################################################
"""
#Dynamic NonLagged vs Dynamic Lagged
print("Experiment 4\n")

FNC = f.functionalNetworkConnectivity()
lagged = 0
measure = 'DC'
windowsSize = 120
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group1 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group2 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='manwhitneyu', outlier=outlier)

FNC = f.functionalNetworkConnectivity()
lagged = 3
measure = 'DC'
windowsSize = 120
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group3 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group4 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
utils.toFindStatisticDifference(utils.buildFeaturesVector(group3), utils.buildFeaturesVector(group4), measure='manwhitneyu', outlier=outlier)
pg.barchart2(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonLagged Static correlation vs Max Lagged Dynamic correlation", labelGroup1="NLSC MCS", labelGroup2="NLSC VS/UWS", labelGroup3="LDC MCS", labelGroup4="LDC VS/UWS", xLabel="# Connections", yLabel="Connectivity Mean", outlier=outlier)
pg.barchart3(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonLagged Static correlation vs Max Lagged Dynamic correlation", labelGroup1="NLSC Mean MCS", labelGroup2="NLSC Mean VS/UWS", labelGroup3="LDC Mean MCS", labelGroup4="LDC Mean VS/UWS", labelGroup5="NLSC Std MCS", labelGroup6="NLSC Std VS/UWS", labelGroup7="LDC Std MCS", labelGroup8="LDC Std VS/UWS", xLabel="", yLabel="Mean Values", outlier=outlier)
"""
#########################################################################################################################

#########################################################################################################################
"""
print("Experiment 5\n")

FNC = f.functionalNetworkConnectivity()
lagged = 0
measure = 'PC'
windowsSize = None
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group1 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group2 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='manwhitneyu', outlier=outlier)

FNC = f.functionalNetworkConnectivity()
lagged = 3
measure = 'PC'
windowsSize = 120
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group3 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group4 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
utils.toFindStatisticDifference(utils.buildFeaturesVector(group3), utils.buildFeaturesVector(group4), measure='manwhitneyu', outlier=outlier)
pg.barchart2(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonLagged Static correlation vs Max Lagged Dynamic correlation", labelGroup1="NLSC MCS", labelGroup2="NLSC VS/UWS", labelGroup3="LDC MCS", labelGroup4="LDC VS/UWS", xLabel="# Connections", yLabel="Connectivity Mean", outlier=outlier)
pg.barchart3(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonLagged Static correlation vs Max Lagged Dynamic correlation", labelGroup1="NLSC Mean MCS", labelGroup2="NLSC Mean VS/UWS", labelGroup3="LDC Mean MCS", labelGroup4="LDC Mean VS/UWS", labelGroup5="NLSC Std MCS", labelGroup6="NLSC Std VS/UWS", labelGroup7="LDC Std MCS", labelGroup8="LDC Std VS/UWS", xLabel="", yLabel="Mean Values", outlier=outlier)
"""
#########################################################################################################################

#########################################################################################################################
"""
# Probar con frecuencia 0.1
# Brain dynamics low frequency

print("Experiment 6\n")

FNC = f.functionalNetworkConnectivity()
lagged = 3
measure = 'PC'
windowsSize = 240
f_lb = 0.008
f_ub = 0.1

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group1 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group2 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='manwhitneyu', outlier=outlier)

FNC = f.functionalNetworkConnectivity()
lagged = 3
measure = 'PC'
windowsSize = 240
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group3 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group4 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
utils.toFindStatisticDifference(utils.buildFeaturesVector(group3), utils.buildFeaturesVector(group4), measure='manwhitneyu', outlier=outlier)
pg.barchart2(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonFiltered vs Filtered", labelGroup1="NF MCS", labelGroup2="NF VS/UWS", labelGroup3="F MCS", labelGroup4="F VS/UWS", xLabel="# Connections", yLabel="Connectivity Mean", outlier=outlier)
pg.barchart3(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonFiltered vs Filtered ", labelGroup1="NF Mean MCS", labelGroup2="NF Mean VS/UWS", labelGroup3="F Mean MCS", labelGroup4="F Mean VS/UWS", labelGroup5="NF Std MCS", labelGroup6="NF Std VS/UWS", labelGroup7="F Std MCS", labelGroup8="F Std VS/UWS", xLabel="", yLabel="Mean Values", outlier=outlier)
"""
#########################################################################################################################


#########################################################################################################################
"""
print("Experiment 7\n")

FNC = f.functionalNetworkConnectivity()
lagged = 3
windowsSize = 240
measure = 'DC'
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group1 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group2 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), measure='manwhitneyu', outlier=outlier)

FNC = f.functionalNetworkConnectivity()
lagged = 3
windowsSize = 240
measure = 'PC'
f_lb = 0.008
f_ub = 0.05

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'
group3 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Vegetative State/'
group4 = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order, wSize=windowsSize, lag=lagged, measure=measure, reduce_neuronal = True, reductionMeasure = 'max', RSN=True)

#cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size = 1000, alpha=0.5, scale = 10, namesNodes=namesNodes_node_to_node)
utils.toFindStatisticDifference(utils.buildFeaturesVector(group3), utils.buildFeaturesVector(group4), measure='manwhitneyu', outlier=outlier)
pg.barchart2(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonLinear FLD correlation vs Linear FLD correlation", labelGroup1="NLFLD MCS", labelGroup2="NLFLD VS/UWS", labelGroup3="LFLD MCS", labelGroup4="LFLD VS/UWS", xLabel="# Connections", yLabel="Connectivity Mean", outlier=outlier)
pg.barchart3(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),  utils.buildFeaturesVector(group3),  utils.buildFeaturesVector(group4), title= "NonLinear FLD correlation vs Linear FLD correlation", labelGroup1="NLFLD Mean MCS", labelGroup2="NLFLD Mean VS/UWS", labelGroup3="LFLD Mean MCS", labelGroup4="LFLD Mean VS/UWS", labelGroup5="NLFLD Std MCS", labelGroup6="NLFLD Std VS/UWS", labelGroup7="LFLD Std MCS", labelGroup8="LFLD Std VS/UWS", xLabel="", yLabel="Mean Values", outlier=outlier)

"""
#########################################################################################################################

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

# Teoria de la conciencia
# FeedBack
# Laggeds

# Experimento 1 -> FNC - NBS
# Experimento 2 -> Poder de Clasificación Diagnostica
# Experimento 3 -> Zero Lagged
# Experimento 4 -> Autocorrelacion en distance correlation
# Experimento 5 -> Efecto de los filtros sobre los niveles de correlacion (Antes y Despues)
