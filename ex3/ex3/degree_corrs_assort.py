# -*- coding: utf-8 -*-
"""
Model code for BECS-114.4150 Complex Networks problem 3.1
(degree correlations and assortativity)
Written by Onerva Korhonen.
Modified by Rainer Kujala (11.1.2016)

Created on Wed Aug 20 14:30:41 2014
Modified to create a template on 13.4.2015

@author: aokorhon
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from ex3.colorbar_help import add_colorbar
from scipy.stats import binned_statistic_2d
import operator


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


def get_x_and_y_degrees(network: nx.Graph) -> (np.array, np.array):
    """
    For the given network, creates two arrays (x_degrees
    and y_degrees) of the degrees of "start" and "end" nodes of each edge in
    the network. For undirected networks, each edge is considered twice.

    Parameters
    ----------
    network: a NetworkX graph object

    Returns
    -------
    degree_arrays: a  (x_degrees, y_degrees) tuple where x_degrees and
    y_degrees are NumPy arrays
    """
    edges = network.edges()
    n_edges = len(edges)

    x_degrees = np.zeros(2 * n_edges)
    y_degrees = np.zeros(2 * n_edges)

    for i, (start, end) in enumerate(network.edges_iter()):
        x_degrees[2 * i] = network.degree(start)
        x_degrees[2 * i + 1] = network.degree(end)

        y_degrees[2 * i] = network.degree(end)
        y_degrees[2 * i + 1] = network.degree(start)

    return x_degrees, y_degrees


def create_scatter(degree_arrays, network_name, network_title, figure_base):
    """
    For x_degrees, y_degrees pair in degree_arrays, creates and
    saves a scatter of the degrees.

    Parameters
    ----------
    degree_arrays : tuple of numpy arrays
        (x_degrees, y_degrees)
    network_name: str
        network name (string) for naming purposes
    network_title: str
        a network-referring title (string) for figures
    figure_base: str
        A base path for the heatmap figures. network_name and .pdf
        extension is added after the figure_base

    Returns
    -------
    no output, but scatter plot (as pdf) is saved into the given path
    """
    assert degree_arrays, "degree_arrays are not properly defined"

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_title(network_title)
    ax.set_xlabel(r'Degree $k$')
    ax.set_ylabel(r'Degree $k$')

    # TODO : add clutering
    ax.set_xlim(0, max(degree_arrays[0]))
    ax.set_ylim(0, max(degree_arrays[1]))

    ax.scatter(x=degree_arrays[0], y=degree_arrays[1], alpha=0.4, s=1)
    plt.savefig(figure_base + network_name + '.pdf', format='pdf', bbox_inches='tight')

    print("Scatter plot ready!")


def create_heatmap(degree_arrays, network_name, network_title, figure_base):
    """
    For x_degrees, y_degrees pair in degree_arrays, creates and
    saves a heatmap of the degrees.

    Parameters
    ----------
    degree_arrays : tuple of numpy arrays
        (x_degrees, y_degrees)
    network_name: str
        network name (string) for naming purposes
    network_title: str
        a network-referring title (string) for figures
    figure_base: str
        A base path for the heatmap figures. network_name and .pdf
        extension is added after the figure_base

    Returns
    -------
    no output, but heatmap figure (as pdf) is saved into the given path
    """
    if degree_arrays:
        x_degrees, y_degrees = degree_arrays
        k_min = np.min(degree_arrays)
        k_max = np.max(degree_arrays)

        n_bins = k_max - k_min + 1
        values = np.zeros(x_degrees.size)
        statistic = binned_statistic_2d(x=x_degrees, y=y_degrees, bins=n_bins,
                                        statistic="count", values=values)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        ax.imshow(statistic.statistic, extent=(k_min - 0.5, k_max + 0.5, k_min - 0.5, k_max + 0.5),
                  origin='lower', cmap='hot', interpolation='nearest')

        ax.set_title(network_title)
        ax.set_xlabel('Degree $k$')
        ax.set_ylabel('Degree $k$')

        add_colorbar(statistic.statistic, cmap='hot')
        plt.savefig(figure_base + network_name + '.pdf', format='pdf', bbox_inches='tight')

        print("Heatmap ready!")
    else:
        print("Network poorly defined, cannot produce heatmap...")


def assortativity(degree_arrays):
    """
    Calculates assortativity for a network, i.e. Pearson correlation
    coefficient between x_degrees and y_degrees in the network.

    Parameters
    ----------
    degree_arrays: tuple of numpy arrays
        (x_degrees, y_degrees)

    Returns
    -------
    assortativity: float
        the assortativity value of the network as a number
    """
    k_x, k_y = degree_arrays[0], degree_arrays[1]

    a = np.mean(k_x * k_y)
    b = np.mean(k_x) * np.mean(k_y)

    c = np.math.sqrt((np.mean(k_x * k_x)) - np.mean(k_x) ** 2)
    d = np.math.sqrt((np.mean(k_y * k_y)) - np.mean(k_y) ** 2)

    return (a - b) / (c * d)


def get_nearest_neighbor_degree(network: nx.graph):
    """
    Calculates the average nearest neighbor degree for each node for the given
    list of networks.

    Parameters
    ----------
    network: a NetworkX graph objects

    Returns
    -------
    degrees: list-like
        an array of node degree
    nearest_neighbor_degrees: list-like
        an array of node average nearest neighbor degree in the same order
        as degrees
    """
    degrees = []
    nearest_neighbor_degrees = []

    deg = nx.degree(network)
    nnd = nx.average_neighbor_degree(network)

    for (key, item) in sorted(deg.items(), key=operator.itemgetter(1)):
        degrees.append(item)
        nearest_neighbor_degrees.append(nnd[key])

    return degrees, nearest_neighbor_degrees


def get_simple_bin_average(x_values, y_values):
    """
    Calculates average of y values within each x bin. The binning used is the
    most simple one: each unique x value is a bin of it's own.

    Parameters
    ----------
    x_values: an array of x values
    y_values: an array of corresponding y values

    Returns
    -------
    bins: an array of unique x values
    bin_average: an array of average y values per each unique x
    """

    bx, by = [], []

    last_x = x_values[0]
    bx.append(last_x)

    temp = []

    for x, y in zip(x_values, y_values):
        if x != last_x:
            last_x = x
            bx.append(x)
            by.append(np.mean(temp))
            temp = [y]
        else:
            temp.append(y)

    by.append(np.mean(temp))
    return np.array(bx), np.array(by)


def visualize_nearest_neighbor_degree(degrees,
                                      nearest_neighbor_degrees,
                                      bins,
                                      bin_averages,
                                      network_name,
                                      network_title,
                                      figure_base
                                      ):
    """
    Visualizes the nearest neighbor degree for each degree as a scatter and
    the mean nearest neighbor degree per degree as a line.

    Parameters
    ----------
    degrees: list-like
        an array of node degrees
    nearest_neighbor_degrees: list-like
        an array of node nearest neighbor degrees in the same order as degrees
    bins: list-like
        unique degree values
    bin_averages: list-like
        the mean nearest neighbor degree per unique degree value
    network_name: str
        network name (string) for naming purposes
    network_title: str
        network-referring title (string) for figure
    figure_base: str
        a base path to the directory where the heatmap figures should be
        saved. Network_name and .pdf extension should be added after the
        figure_base variable.

    Returns
    -------
    Nothing. Figure (as pdf) should be saved to the given path
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_title(network_title)
    ax.set_xlabel('Degree $k$')
    ax.set_ylabel('Nearest Neighbor Degrees $k_{nn}(n)$')

    ax.set_xlim(0, max(degrees))
    ax.set_ylim(0, max(nearest_neighbor_degrees))

    ax.scatter(x=degrees, y=nearest_neighbor_degrees, c="b", s=3, label="$k_{nn}(n)$")
    ax.plot(bins, bin_averages, "r--", label="$< k_{nn} > (n)$")
    ax.legend(loc="best")

    plt.savefig(figure_base + network_name + '.pdf', format='pdf', bbox_inches='tight')


def main():
    network_paths = ["facebook-wosn-links_subgraph.edg",
                     "karate_club_network_edge_file.edg"]

    network_names = ["facebook wosn", "karate_club_network"]
    network_titles = ["facebook wosn", "karate club network"]

    scatter_figure_base = 'graphs/scatter'
    heatmap_figure_base = 'graphs/heatmap'
    nearest_neighbor_figure_base = 'graphs/nearest_neighbors'

    for network_path, network_name, network_title in zip(network_paths, network_names, network_titles):
        network = read_network(network_path)
        degree_arrays = get_x_and_y_degrees(network)

        create_scatter(degree_arrays, network_name, network_title, scatter_figure_base)
        create_heatmap(degree_arrays, network_name, network_title, heatmap_figure_base)

        assortativity_own = assortativity(degree_arrays)
        assortativity_nx = nx.degree_assortativity_coefficient(network)

        print("Own assortativity for " + network_title + ": " + str(assortativity_own))
        print("NetworkX assortativity for " + network_title + ": " + str(assortativity_nx))

        degrees, nearest_neighbor_degrees, = get_nearest_neighbor_degree(network)

        unique_degrees, mean_nearest_neighbor_degrees = get_simple_bin_average(degrees, nearest_neighbor_degrees)

        visualize_nearest_neighbor_degree(
            degrees,
            nearest_neighbor_degrees,
            unique_degrees,
            mean_nearest_neighbor_degrees,
            network_name,
            network_title,
            nearest_neighbor_figure_base)


if __name__ == '__main__':
    main()
