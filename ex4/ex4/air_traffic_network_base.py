import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

GRAPHS_PATH = "graphs/ex3/"


def plot_network_usa(net, xycoords, edges=None):
    """
    Plot the network usa.

    Parameters
    ----------
    net : the network to be plotted
    xycoords : dictionary of node_id to coordinates (x,y)
    edges : list of node index tuples (node_i,node_j),
            if None all network edges are plotted.
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 0.9])
    # ([0, 0, 1, 1])
    bg_figname = 'US_air_bg.png'
    img = plt.imread(bg_figname)
    axis_extent = (-6674391.856090588, 4922626.076444283,
                   -2028869.260519173, 4658558.416671531)
    ax.imshow(img, extent=axis_extent)
    ax.set_xlim((axis_extent[0], axis_extent[1]))
    ax.set_ylim((axis_extent[2], axis_extent[3]))
    ax.set_axis_off()
    nx.draw_networkx(net,
                     pos=xycoords,
                     with_labels=False,
                     node_color='k',
                     node_size=5,
                     edge_color='r',
                     alpha=0.2,
                     edgelist=edges)
    return fig, ax


def get_properties(network: nx.Graph):
    N = len(network)
    L = len(list(network.edges()))
    D = nx.density(network)
    d = nx.diameter(network)
    C = nx.average_clustering(network, weight="weight")
    return N, L, D, d, C


def plot_max_spanning_tree(net: nx.Graph, xycoords):
    mst = nx.maximum_spanning_tree(net)
    fig, ax = plot_network_usa(mst, xycoords)
    fig.savefig("{}airports_max_spanning_tree.pdf".format(GRAPHS_PATH))
    return mst


def plot_min_spanning_tree(net: nx.Graph, xycoords):
    mst = nx.minimum_spanning_tree(net)
    print(len(mst))
    fig, ax = plot_network_usa(mst, xycoords)
    fig.savefig("{}airports_min_spanning_tree.pdf".format(GRAPHS_PATH))
    return mst


def plot_threshold_network(M, net, xycoords):
    edges = set((u, v) for (u, v, _) in reversed(sorted(net.edges(data=True), key=lambda tup: tup[2]["weight"])))
    edges_to_keep = set(list((u, v) for (u, v, _) in
                             reversed(sorted(net.edges(data=True), key=lambda tup: tup[2]["weight"])))[:M])
    net.remove_edges_from(edges - edges_to_keep)
    fig, ax = plot_network_usa(net, xycoords)
    fig.savefig("{}airports_threshold.pdf".format(GRAPHS_PATH))
    return net


def main():
    id_data = np.genfromtxt('US_airport_id_info.csv', delimiter=',', dtype=None, names=True)
    xycoords = {}

    for row in id_data:
        xycoords[str(row['id'])] = (row['xcoordviz'], row['ycoordviz'])

    net = nx.read_weighted_edgelist("aggregated_US_air_traffic_network_undir.edg")

    # network properties
    print(get_properties(net))

    # display full network
    fig, ax = plot_network_usa(net, xycoords)
    fig.savefig("{}airports.pdf".format(GRAPHS_PATH))

    # plot spanning trees
    mst = plot_max_spanning_tree(net, xycoords)
    plot_min_spanning_tree(net, xycoords)

    # Threshold network
    threshold_net = plot_threshold_network(len(mst), net.copy(), xycoords)

    edges_mst = set(tuple(sorted((u, v))) for (u, v) in mst.edges())
    edges_threshold = set(tuple(sorted((u, v))) for (u, v) in threshold_net.edges())
    edges_commun = edges_mst.intersection(edges_threshold)
    print(len(edges_commun))
    print(edges_commun)


if __name__ is "__main__":
    main()
