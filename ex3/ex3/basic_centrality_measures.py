# -*- coding: utf-8 -*-
"""
Python coding template for CS-E5740 Complex Networks problem 3.3 (Basic
centrality measures).
Written by Onerva Korhonen.

Original example code created on Mon Aug 25 10:23:54 2014.
Modified to create template 20.10.2015.
Template updated 19.10.2016.

@author: aokorhon
"""
import numpy as np
import networkx as nx
import matplotlib.pylab as plt
from matplotlib import gridspec
from ex3.colorbar_help import add_colorbar
import pickle


def read_network(network_path: str) -> nx.graph:
    """
    Reads a given network.

    Parameters
    ----------
    network_path: network edge list path (string)

    Returns
    -------
    network: NetworkX weighted graph object
    """
    return nx.read_weighted_edgelist(path=network_path)


def read_unweighted_network(network_path: str) -> nx.graph:
    """
    Reads a given network.

    Parameters
    ----------
    network_path: network edge list path (string)

    Returns
    -------
    network: NetworkX graph object
    """
    return nx.read_edgelist(path=network_path)


def get_centrality_measures(network: nx.Graph, tol: float):
    """
    Calculates five centrality measures (degree, betweenness, closeness, and
    eigenvector centrality, and k-shell) for the nodes of the given network.

    use NetworkX functions to obtain centrality measure dictionaries
    sort the dictionary values into arrays in the order given by
    network.nodes().
    Hint: make use of get method of dictionaries.

    Parameters
    ----------
    network: networkx.Graph()
    tol: tolerance parameter to calculate eigenvector centrality, float

    Returns
    --------
    [degree, betweenness, closeness, eigenvector_centrality, kshell]: list of
    lists
    """

    degree, betweenness, closeness, eigenvector_centrality, kshell = [], [], [], [], []
    betweenness_dict = nx.betweenness_centrality(network)
    closeness_dict = nx.closeness_centrality(network)
    eigenvector_centrality_dict = nx.eigenvector_centrality(network, tol=tol)
    kshell_dict = nx.core_number(network)

    for n in network.nodes_iter():
        degree.append(network.degree(n))
        betweenness.append(betweenness_dict[n])
        closeness.append(closeness_dict[n])
        eigenvector_centrality.append(eigenvector_centrality_dict[n])
        kshell.append(kshell_dict[n])

    return [degree, betweenness, closeness, eigenvector_centrality, kshell]


def create_scatter(x_values, y_values, x_label, y_label, labels, markers, fig, figure_path: str):
    """
    Creates a scatter plot of y_values as a function of x_values. 

    Note that y_values is a list of lists and this function plots the y_values of each list
    against the same x_values.

    Parameters
    ----------
    x_values: np.array
    y_values: list of lists
    x_label: string
    y_label: string
        a generic label of the y axis
    labels: list of strings
        labels of scatter plots
    markers: list of strings
    figure_path: string

    Returns
    -------
    No direct output, saves the scatter plot at given figure_path
    """
    assert len(x_values) > 0, 'Bad input x_values for creating a scatter plot'

    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for (y_value, label, marker) in zip(y_values, labels, markers):
        ax.scatter(x=x_values, y=y_value, label=label, marker=marker, s=40)

    ax.legend(loc='best')
    plt.savefig(figure_path, bbox_inches='tight')
    print('Scatter plot ready!')


def visualize_on_network(network, node_values, coords_path, fig,
                         titles, figure_path, cmap='OrRd',
                         node_size=50, font_size=8, scale=500):
    """
    Creates visualizations of the network with nodes color coded by each of the
    node values sets.

    Parameters
    ----------
    network: networkx.Graph()
    node_values: list of lists
    coords_path: path to a file containing node coordinates
    fig: matplotlib.pyplot.figure()
    titles: list of strings
    figure_path: string
    cmap: string
    node_size: int
    font_size: int
    scale: int
        used to calculate the spring layout for node positions

    Returns
    -------
    No direct output, saves the network visualizations at given path
    """
    assert len(node_values[0]) > 0, "there should be multiple values per node"

    # This is the grid for 5 pictures
    gs = gridspec.GridSpec(3, 4, width_ratios=(20, 1, 20, 1))
    network_gs_indices = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0)]
    cbar_gs_indices = [(0, 1), (0, 3), (1, 1), (1, 3), (2, 1)]

    # Loading coordinates from the file
    with open(coords_path, 'rb') as f:
        coords = pickle.load(f, encoding='latin1')

    # Loop over different value sets
    for node_val, title, network_gs_index, cb_gs_index in zip(node_values,
                                                              titles,
                                                              network_gs_indices,
                                                              cbar_gs_indices):
        # Draw the network figure
        ax = plt.subplot(gs[network_gs_index[0], network_gs_index[1]])
        nx.draw(network, pos=coords, node_color=node_val, cmap=cmap,
                node_size=int(node_size), font_size=font_size)

        # Draw the colorbar (cb)
        cb_ax = plt.subplot(gs[cb_gs_index[0], cb_gs_index[1]])
        add_colorbar(node_val, cb_ax=cb_ax)

        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(figure_path, format='pdf', bbox_inches='tight')
    print('Network visualizations ready!')


def main():
    KARATE = "karate_club_network"

    network_paths = [
        "small_cayley_tree.edg",
        "small_lattice.edg",
        "small_ring.edg",
        "karate_club_network_edge_file.edg"
    ]

    coords_paths = [
        'small_cayley_tree_coords.pkl',
        'small_lattice_coords.pkl',
        'small_ring_coords.pkl',
        'karate_club_coords.pkl'
    ]

    network_names = ['small_cayley_tree', 'lattice', 'small_ring', KARATE]
    x_label = 'degree'
    y_label = 'centrality measure'

    labels = ['betweenness', 'closeness', 'k-shell', 'eigenvector centrality']
    markers = ['.', 'x', '+', 'o']
    scatter_base_path = 'graphs/centrality/'
    titles = ['small cayley tree', 'larger lattice', 'small ring', 'karate club network']
    network_fig_base_path = scatter_base_path
    fig_index = 0
    tol = 10 ** -1  # tolerance parameter for calculating eigenvector centrality

    # Loop through all networks
    for (network_path, network_name, coords_path) in zip(network_paths, network_names, coords_paths):
        network = \
            (read_network(network_path) if network_name == KARATE else read_unweighted_network(network_path))

        # Calculating centrality measures
        [degree, betweenness, closeness, eigenvector_centrality, kshell] = get_centrality_measures(network, tol=tol)
        kshell_normalized = np.divide(kshell, float(max(kshell)))  # normalization for easier visualization

        # Scatter plot
        y_values = [betweenness, closeness, eigenvector_centrality, kshell_normalized]
        scatter_path = scatter_base_path + '_' + network_name + '.pdf'
        scatter_fig = plt.figure(fig_index)
        fig_index = fig_index + 1

        create_scatter(
            x_values=degree,
            y_values=y_values,
            x_label=x_label,
            y_label=y_label,
            labels=labels,
            markers=markers,
            fig=scatter_fig,
            figure_path=scatter_path,
        )

        # Network figures
        network_fig = plt.figure(fig_index)
        fig_index = fig_index + 1
        network_figure_path = network_fig_base_path + '_networks_' + network_name + '.pdf'
        all_cvalues = [degree, betweenness, closeness, eigenvector_centrality, kshell_normalized]

        visualize_on_network(
            network=network,
            node_values=all_cvalues,
            coords_path=coords_path,
            fig=network_fig,
            titles=["degree", "betweenness", "closeness", "eigenvector_centrality", "kshell"],
            figure_path=network_figure_path
        )


if __name__ == '__main__':
    main()
