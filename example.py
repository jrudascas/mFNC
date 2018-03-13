import functionalNetworkConnectivity as f
import numpy as np
from nilearn import plotting as plott

f_order = 2
outlier = -1.1
f_lb = 0.008
f_ub = 0.05
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
namesNodes_node_to_node = ['Auditory', 'Cerebellum', 'DMN', 'ECL', 'ECR', 'Salience', 'SensoriMotor', 'Vis_Lateral', 'Vis_Medial', 'Vis_Occipital']
TR = 2                      # Repetition time
windowsSize = None          # 240sec
lagged = 0                  # -6sec to 6sec
measure = 'PC'              # DC or PC
reduce_neuronal = True      # if reduce_neuronal is True will be used the neuronal classifier
reductionMeasure = 'max'    # measure used in the reductión of the lag and dynamic version of FNC
onlyRSN = True              # if onlyRSN is True will be only used the independet components assigned to a RSN
                            # if onlyRSN is False will be all independents components coming from ICA

FNC = f.functionalNetworkConnectivity()

path = '/home/jrudascas/Desktop/Projects/Dataset/Original/Minimal Conscience/'

correlationMatrix, lagsProbabilityDistribution = FNC.run(path=path, TR=TR, f_lb=f_lb, f_ub=f_ub, f_order=f_order,
                                                         wSize=windowsSize, lag=lagged, measure=measure,
                                                         reduce_neuronal=reduce_neuronal,
                                                         reductionMeasure=reductionMeasure, onlyRSN=True)

correlationMatrix = FNC.reduce_node_to_node_connectivity(correlationMatrix, outlier=outlier, mandatory = True)
np.fill_diagonal(correlationMatrix, 0)

plott.plot_matrix(correlationMatrix, labels=namesNodes_node_to_node, vmax=0.6, vmin=-0.6)
plott.plot_connectome(correlationMatrix, coords, node_color=['b', 'r', 'g', 'gold', 'y', 'aqua', 'orchid', 'lime', 'black', 'brown'],
                      node_size=200, edge_vmax=.6, title='MCS', edge_vmin=-0.6, colorbar=True)

plott.show()

