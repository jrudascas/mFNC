import community
import networkx as nx
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm
import numpy as np
import forceatlas as fa

def buildG_from_file(file_, delimiter_):
    G = nx.Graph()
    reader = csv.reader(open(file_), delimiter=delimiter_)
    for line in reader:
        G.add_edge(int(line[0]),int(line[1]), weight=float(line[2]))
    return G

def buildG_from_adjacency_matrix(data, labels=None, allNode = False):
    G = nx.Graph()
    size = range(data.shape[-1])
    for index1 in size:
        for index2 in size:
            if index2 > index1:
                if abs(data[index1, index2]) != 0.0:
                    if labels is None:
                        G.add_edge(index1, index2, weight=abs(data[index1, index2]))
                    else:
                        G.add_edge(labels[index1], labels[index2], weight=abs(data[index1, index2]))
                elif allNode:
                    if labels is None:
                        G.add_node(index1)
                        G.add_node(index2)
                    else:
                        G.add_node(labels[index1])
                        G.add_node(labels[index2])

    return G

def circular_layout2(G, scale=1., center=None):

    import numpy as np

    if len(G) == 0:
        return {}

    twopi = 2.0*np.pi
    theta = np.arange(0, twopi, twopi/len(G))
    theta = -1*theta + np.pi/2

    pos = np.column_stack([np.cos(theta), np.sin(theta)]) * scale
    if center is not None:
        pos += np.asarray(center)

    return dict(zip(G, pos))

def draw_graph(G, node_size, alpha, scale = 10, namesNodes=None, returnPlot = False):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(G))))

    pos = circular_layout2(G)

    #pos = fa.forceatlas2_layout(G, linlog=True, nohubs=False, iterations=500)  # compute graph layout
    # plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=node_size, cmap=plt.cm.RdYlBu, node_color=list(colors))

    edgewidth = [d['weight']*scale for (u, v, d) in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, alpha=alpha, width=edgewidth)

    dict = {}

    if namesNodes is None:
        for node, l in G.node.items():
            dict[node] = str(node)
    else:
        namesNodes = iter(namesNodes)
        for node, l in G.node.items():
            dict[node] = next(namesNodes)

    nx.draw_networkx_labels(G, pos, dict, font_size=8)

    if returnPlot is False:
        plt.show()
    else:
        return plt

def find_best_partition(path=None, delimiter = None, adjacencyMatrix=None, labels = None):

    if path is not None and delimiter is not None:

        G = buildG_from_file(path, delimiter)

    if (adjacencyMatrix is not None):

        G = buildG_from_adjacency_matrix(adjacencyMatrix, labels)

    partition = community.best_partition(G, resolution=0.8)

    size = int(len(set(partition.values())))

    #position = nx.fruchterman_reingold_layout(G)

    """

    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        nx.draw_networkx_nodes(G, position, list_nodes, node_size = 20, node_color = next(colors), alpha=0.8)
        count = count + 1

    nx.draw_networkx_edges(G, position, alpha=0.5)
    plt.show()
    """
    """
    pos = fa.forceatlas2_layout(G, linlog=False, nohubs=False, iterations=500) # compute graph layout
    plt.figure(figsize=(7, 7))  # image is 8 x 8 inches
    plt.axis('off')

    #print("Comunnities Number " + str(size))

    colors = ['red', 'blue', 'green', 'purple', 'gray', 'yellow', 'c',  'palegreen', 'darkorchid', 'darkblue', 'crimson',
              'pink', 'teal', 'brown', 'coral', 'olive', 'orangered', 'lime', 'mediumpurple', 'azure', 'beige', 'crimson',
              'khaki', 'lavender', 'salmon', 'tan', 'yellogreen']
    #colors = range(size)
    #t = np.linspace(0, TimeCourse.shape[0], TimeCourse.shape[0], endpoint=False)
    nodeColors = []

    for node in G.nodes():
        nodeColors.append(colors[partition.get(node)])

    #print("Nodes Number " + str(len(nodeColors)))

    nx.draw_networkx_nodes(G, pos, node_size=350, cmap=plt.cm.RdYlBu, node_color=nodeColors, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    for c in range(size):
        edgesList = []
        for edge in G.edges():
            if partition.get(edge[0]) == partition.get(edge[1]) and partition.get(edge[1]) == c:
                edgesList.append(edge)
        nx.draw_networkx_edges(G, pos, edge_cmap=plt.cm.RdYlBu, edgelist=edgesList, edge_color=colors[int(c)], width=2.0, alpha=0.5)

    dict = {}
    for node, l in G.node.items():
        dict[node] = str(node)

    nx.draw_networkx_labels(G, pos, dict, font_size=10)
    """
    #plt.show()
    return partition