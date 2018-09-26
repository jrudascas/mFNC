import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.ioff()

import core as core
import utils
import sys
import plotGallery as pg
import numpy as np
from nilearn import plotting as plott
# np.set_printoptions(precision=3)
# np.set_printoptions(threshold=np.inf)

namesNodes_node_to_node = ['Auditory', 'Cerebellum', 'DMN', 'ECL', 'ECR', 'Salience', 'SensoriMotor', 'Vis_Lateral',
                           'Vis_Medial', 'Vis_Occipital']
# namesNodes_node_to_node = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '1O']
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

TR = 2.46
outlier = -1.1
umbral = 0.6
lagged = 3
windowsSize = None
path_general = '/home/jrudascas/Desktop/Projects/Dataset/Original'
#path_general = '/home/runlab/data/COMA_ICA'
core = core.Core()

group1, laggeds1, TD1, AWTD1 = core.run(path=path_general + '/MCS/', TR=TR, wSize=windowsSize, lag=lagged, reduce_neuronal=True, onlyRSN=True)
group2, laggeds2, TD2, AWTD2 = core.run(path=path_general + '/UWS/', TR=TR, wSize=windowsSize, lag=lagged, reduce_neuronal=True, onlyRSN=True)
group3, laggeds3, TD3, AWTD3 = core.run(path=path_general + '/Control/', TR=TR, wSize=windowsSize, lag=lagged, reduce_neuronal=True, onlyRSN=True)

np.savetxt(path_general + '/TD_mcs.out', utils.mean(TD1, outlier=outlier), delimiter=' ', fmt='%s')
np.savetxt(path_general + '/TD_uws.out', laggeds2, delimiter=' ', fmt='%s')
np.savetxt(path_general + '/TD_hc.out', laggeds3, delimiter=' ', fmt='%s')

#np.savetxt('/home/runlab/data/COMA_ICA/' + 'AWTDmcs.out', AWTD1, delimiter=' ', fmt='%s')
#np.savetxt('/home/runlab/data/COMA_ICA/' + 'AWTDuws.out', AWTD2, delimiter=' ', fmt='%s')
#np.savetxt('/home/runlab/data/COMA_ICA/' + 'AWTDhc.out', AWTD3, delimiter=' ', fmt='%s')

#pg.fivethirtyeightPlot(laggeds1, laggeds2, group3=laggeds3, lag=lagged, save='ThreadsLagPC.png')

#print("\nLaggeds MCS UWS")
#pList1 = utils.toFindStatisticDifference(utils.mean(TD1, outlier=outlier), utils.mean(TD2, outlier=outlier),
#                                         measure='manwhitneyu', outlier=outlier)

#print("\nLaggeds MCS HC")
#pList1 = utils.toFindStatisticDifference(utils.mean(TD1, outlier=outlier), utils.mean(TD3, outlier=outlier),
#                                         measure='manwhitneyu', outlier=outlier)

#print("\nLaggeds UWS HC")
#pList1 = utils.toFindStatisticDifference(utils.mean(TD2, outlier=outlier), utils.mean(TD3, outlier=outlier),
#                                         measure='manwhitneyu', outlier=outlier)


print("\nTest Graph MCS UWS")
pList1 = utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2),
                                         measure='manwhitneyu', outlier=outlier, is_corrected=True)

#print("\nTest Graph HC MCS")
#pList1 = utils.toFindStatisticDifference(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group3),
#                                         measure='manwhitneyu', outlier=outlier, is_corrected=True)

#print("\nTest Graph HC UWS")
#pList1 = utils.toFindStatisticDifference(utils.buildFeaturesVector(group2), utils.buildFeaturesVector(group3),
#                                         measure='manwhitneyu', outlier=outlier, is_corrected=True)

sys.exit(0)

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
                new3[i, j] = len(np.where(abs(auxTemp1) > umbral)[0])
            # new[i, j] = np.mean(aux)

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
                new4[i, j] = len(np.where(abs(auxTemp2) > umbral)[0])

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

# cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group1, outlier=-1.1), allNode=True), node_size=2000, alpha=0.5, scale=20, namesNodes=namesNodes_node_to_node, save="MCS1")
# cm.draw_graph(cm.buildG_from_adjacency_matrix(FNC.reduce_node_to_node_connectivity(group2, outlier=-1.1), allNode=True), node_size=2000, alpha=0.5, scale=20, namesNodes=namesNodes_node_to_node, save="VS1")
# pg.barchart(utils.buildFeaturesVector(group1), utils.buildFeaturesVector(group2), title= "Non-Lagged Linear Correlation", labelGroup1="MCS", labelGroup2="VS/UWS", xLabel="", yLabel="Average Connectivity Level", outlier=outlier, save="fig11.png", labelFeautures=namesNodes_edge_to_edge)

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

correlation_matrix1 = core.reduce_node_to_node_connectivity(group1, outlier=outlier, mandatory=True)
correlation_matrix2 = core.reduce_node_to_node_connectivity(group2, outlier=outlier, mandatory=True)
correlation_matrix3 = core.reduce_node_to_node_connectivity(group3, outlier=outlier, mandatory=True)

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
        new = new / 45

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
        new = new / 45

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
        new = new / 45

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
                      node_color=['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                      node_size=200, edge_vmax=.8, title='MCS', edge_vmin=-0.8, colorbar=True)

plott.plot_connectome(correlation_matrix2,
                      coords,
                      node_color=['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                      node_size=200, edge_vmax=.8, title='UWS', edge_vmin=-0.8, colorbar=True)

plott.plot_connectome(correlation_matrix3,
                      coords,
                      node_color=['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                      node_size=200, edge_vmax=.8, title='HC', edge_vmin=-0.8, colorbar=True)

# plotting.plot_connectome(correlation_matrix1,
#                         coords, edge_threshold='90%',
#                         title=title,
#                         edge_vmax=.9, edge_vmin=-.9)

plott.plot_matrix(correlation_matrix1, labels=namesNodes_node_to_node, vmax=0.8, vmin=-0.8)
plott.plot_matrix(correlation_matrix2, labels=namesNodes_node_to_node, vmax=0.8, vmin=-0.8)
plott.plot_matrix(correlation_matrix3, labels=namesNodes_node_to_node, vmax=0.8, vmin=-0.8)
plott.show()

sys.exit(0)
