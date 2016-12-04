from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_distribution(y_values: np.ndarray, x_values: np.ndarray, style, x_label, y_label, figure_file_name):
    """
    Visualizes the pre-calculated distribution y(x) where x is a range from
    min_range to max_range.

    Parameters
    ----------
    y_values: list
        list of values corresponding to the pre-calculated distribution y(x)
    x_values: list
        the x values for plotting
    style: str
        style of the visualization ('bar' or 'logplot')
    x_label: str
        label of the x axis of the figure
    y_label: str
        label of the y axis of the figure
    figure_file_name: str
        full or relative path (including the file name) to save the figure
    """
    if not figure_file_name == '':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if style == 'bar':  # Creates a bar plot
            ax.bar(np.array(x_values) - 0.5, y_values, width=1.0)
        elif style == 'logplot':  # Cretes a loglog xy plot
            ax.loglog(x_values, y_values, 'k', marker='.')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.savefig(figure_file_name, ftype='pdf')


def visualize_network(graph, figure_file_name, title):
    """
    visualizing it with NetworkX and saving the figure into .pdf file.

    Parameters
    ----------
    graph: a networkx Graph object
    figure_file_name: full or relative path (including file name) where to save the figure
    title: title of the figure

    Visualization is done only if a proper figure path is given
    """

    if figure_file_name is not '':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        nx.draw(graph)
        ax.set_title(title)
        fig.savefig(figure_file_name, ftype='pdf')
    return graph


def clustering_and_average_clustering(graph: nx.Graph) -> (List[float], float):
    """
    Returns the clustering coefficient of each node of network and the
    average clustering coefficient.

    Parameters
    ----------
    graph: a NetworkX graph object

    Returns
    -------
    clustering: list
        clustering coefficient of each network node in a

    average_clustering: floating point number
        average clustering coefficient
    """
    clustering = []

    for node in graph.nodes_iter():
        neighbors = nx.neighbors(G=graph, n=node)
        clustering_coef = 0
        n_links = 0

        if len(neighbors) > 1:
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if graph.has_edge(u=neighbor1, v=neighbor2):
                        n_links += 1
            clustering_coef = n_links / (len(neighbors) * (len(neighbors) - 1))
        clustering.append(clustering_coef)

    average_clustering = sum(clustering) / float(len(clustering))

    return clustering, average_clustering


def density(graph: nx.Graph) -> float:
    """
    Calculates the network edge density: D = 2*m / n(n-1)

    Parameters
    ----------
    graph: a NetworkX graph object

    Returns
    -------
    D: network edge density
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    return 0.0 if m == 0 or n <= 1 else (m * 2.0 / float(n * (n - 1)))


def get_degrees(graph: nx.Graph) -> List[int]:
    """
    Returns the degree of each node of network.

    Parameters
    ----------
    graph: a NetworkX graph object

    Returns
    -------
    degrees: list
        degrees of all network node
    """
    deg = []

    for node in graph.nodes_iter():
        deg.append(len(graph.neighbors(n=node)))

    return deg


def calculate_and_visualize_discrete_distribution(input_list, figure_file_name, x_label, y_label):
    """
    Calculates and visualizes the discrete distribution of a variable, the
    values of which are given in input_list and saves the visualization as a
    .pdf file.

    Parameters
    ----------
    input_list: list
        a list of the variable values, e.g. node degrees
    figure_file_name: str
        full or relative path (including file name) where to save the figure
    x_label: str
        label of the x axis of the figure
    y_label: str
        label of the y axis of the figure

    Returns
    -------
    Nothing
    """
    assert len(input_list) > 0, "The input list should not be empty!"

    # Calculate the distribution:
    distribution = np.bincount(input_list)
    n = len(input_list)

    # Normalize:
    distribution = distribution / float(n)

    # Visualize:
    min_range = 0
    max_range = max(input_list) + 1
    x_values = range(min_range, max_range)
    visualize_distribution(distribution, x_values, 'bar', x_label, y_label, figure_file_name)


def cdf(input_list):
    """
    Calculates the cumulative distribution function of list. Note: this is a
    nonstandard version of CDF: CDF(k) is the probability that a random
    variable k has value less than k.

    Parameters
    ----------
    input_list : list
        a sequence of numbers

    Returns
    -------
    cdf: np.array
        cdf of the input list (np.array)
    """
    input_array = np.array(input_list)
    x_points = np.unique(input_array)
    cdfun = np.zeros(len(x_points))

    for i, x in enumerate(x_points):
        cdfun[i] = input_array[np.where(input_array < x)].size

    return cdfun / float(input_array.size)


def create_degree_clustering_scatter(deg, clustering, x_label, y_label, figure_file_name):
    """
    Creates and saves a scatter plot of node clustering coefficient as a
    functionof node degree.

    Parameters
    ----------
    deg : list
        list of node degrees
    clustering : list
        list of node clustering coefficients in the same order as degrees
    x_label : str
        label of the x axis of the figure
    y_label : label of the y axis of the figure
    figure_file_name : full or relative path (including file name) where
                  to save the figure

    Returns
    -------
    Nothing.
    """
    assert figure_file_name != '', "figure path should not be empty"

    # Insert clutter to avoid overlapping points
    n_nodes = len(deg)

    # The values (0.15, and 0.02) don't contain any deeper meaning.
    clutter_degrees = np.random.uniform(-0.15, 0.15, size=n_nodes)
    clutter_clustering = np.random.uniform(-0.02, 0.02, size=n_nodes)
    deg = deg + clutter_degrees
    clustering = clustering + clutter_clustering

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=deg, y=clustering, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(figure_file_name)


def load_network(network_file_name: str):
    """
    A function for loading a network from an edgelist (.edg) file.

    Parameters
    ----------
    network_file_name: full or relative path (including file name) of the .edg file

    Returns
    -------
    network: the loaded network as NetworkX Graph() object
    """
    graph = nx.read_weighted_edgelist(path=network_file_name)
    assert graph is not None, "network was not correctly loaded"
    assert len(graph) > 0, "network should contain at least one node"

    return graph


if '__main__' == __name__:
    # Problem 1.2a)
    network = load_network('karate_club_network_edge_file.edg')
    visualize_network(graph=network, figure_file_name='network.pdf', title='karate club network')

    # Problem 1.2b):
    print('D from self-written algorithm: {0}'.format(str(density(graph=network))))
    print('D from NetworkX function: {0}'.format(str(nx.density(G=network))))

    # Problem 1.2c):
    c_local_own, c_average_own = clustering_and_average_clustering(network)
    c_nx = nx.average_clustering(G=network, count_zeros=True)

    # The parameter count_zeros is set to True to include nodes with C=0 into the average:
    print('C from self-written algorithm: {0}'.format(str(c_average_own)))
    print('C from NetworkX function: {0}'.format(str(c_nx)))

    # Problem 1.2d)
    degrees = get_degrees(graph=network)
    degree_distribution = calculate_and_visualize_discrete_distribution(
        input_list=degrees,
        figure_file_name="degree_distribution.pdf",
        x_label="degrees",
        y_label="proportion"
    )

    cdf = cdf(degrees)
    ccdf_x_values = np.unique(degrees)
    visualize_distribution(
        y_values=1 - cdf,
        x_values=ccdf_x_values,
        style='logplot',
        x_label="degrees",
        y_label="cdf",
        figure_file_name="ccdf.pdf"
    )

    # Problem 1.2e)
    l_nx = nx.average_shortest_path_length(G=network)
    assert l_nx is not None, "Avg. path length has not been computed"
    print('<l> from NetworkX function: {0}'.format(str(l_nx)))

    # Problem 1.2f)
    create_degree_clustering_scatter(
        deg=degrees,
        clustering=c_local_own,
        x_label="degrees",
        y_label="clustering",
        figure_file_name="degree_clustering.pdf"
    )

    plt.show()
