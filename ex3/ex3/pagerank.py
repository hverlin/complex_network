# -*- coding: utf-8 -*-
"""
Programming template for CS-E5740 Complex Networks problem 3.3 (PageRank)
Written by Onerva Korhonen

Created on Tue Sep  8 10:25:49 2015

@author: aokorhon
"""
from random import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import timeit


def page_rank(network: nx.DiGraph, d: float, n_steps: int) -> dict:
    """
    Returns the PageRank value, calculated using a random walker, of each
    node in the network. The random walker teleports to a random node with
    probability 1-d and with probability d picks one of the neighbors of the
    current node.

    Parameters
    -----------
    network : a networkx graph object
    d : damping factor of the simulation
    n_steps : number of steps of the random walker

    Returns
    --------
    page_rank: dictionary of node PageRank values (fraction of time spent in
               each node)
    """
    assert n_steps > 0

    # Initialize PageRank of of each node to 0
    pr, nodes = {}, network.nodes()
    for node in nodes:
        pr[node] = 0

    # Pick a random starting point for the random walker
    curr_node = np.random.choice(nodes)

    # Random walker steps
    for i in range(n_steps):
        # Increase the PageRank of current node by 1
        pr[curr_node] += 1

        sucessors = network.successors(curr_node)

        # Pick the next node either randomly or from the neighbors
        if len(sucessors) == 0 or random() > d:
            curr_node = np.random.choice(nodes)
        else:
            curr_node = np.random.choice(sucessors)

    # Normalize PageRank by n_steps
    for (key, item) in pr.items():
        pr[key] /= n_steps

    return pr


def pagerank_poweriter(g: nx.DiGraph, d: float, iterations: int) -> dict:
    """
    Uses the power iteration method to calculate PageRank value for each node
    in the network.
    
    Parameters
    -----------
    g : a networkx graph object
    d : damping factor of the simulation
    n_iterations : number of iterations to perform
    
    Returns
    --------
    pageRank : dict where keys are nodes and values are PageRank values
    :param iterations:
    """

    # Create a PageRank dictionary and initialize the PageRank of each node to 1/n.
    pr, n = {}, len(g)
    for node in g:
        pr[node] = 1 / n

    nodes = g.nodes()

    for i in range(iterations):
        next_pr = {}
        for key in nodes:
            # Update each node's PageRank to (1-d)*1/n + d*sum(x_j(t-1)/k_j^out).
            next_pr[key] = ((1 - d) / n) + d * sum((pr[x] / g.out_degree(x)) for x in g.predecessors_iter(key))

        pr = next_pr

    return pr


def add_colorbar(cvalues,
                 cmap='OrRd',
                 cb_ax=None):
    """
    Add a colorbar to the axes.

    Parameters
    ----------
    cvalues : 1D array of floats

    """
    import matplotlib as mpl
    eps = np.maximum(0.0000000001, np.min(cvalues) / 1000.)
    vmin = np.min(cvalues) - eps
    vmax = np.max(cvalues)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scm = mpl.cm.ScalarMappable(norm, cmap)
    scm.set_array(cvalues)
    if cb_ax is None:
        plt.colorbar(scm)
    else:
        cb = mpl.colorbar.ColorbarBase(cb_ax,
                                       cmap=cmap,
                                       norm=norm,
                                       orientation='vertical')


def visualize_network(network,
                      node_positions,
                      figure_path,
                      cmap='OrRd',
                      node_size=3000,
                      node_colors=None,
                      w_labels=True,
                      title=""):
    """
    Visualizes the given network using networkx.draw and saves it to the given
    path.

    Parameters
    ----------
    network : a networkx graph object
    node_positions : a list positions of nodes, obtained by e.g. networkx.graphviz_layout
    figure_path : string
    cmap : colormap
    node_size : int
    node_colors : a list of node colors
    w_labels : should node labels be drawn or not, boolean
    title: title of the figure, string
    """

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    if node_colors:
        pos = nx.spring_layout(network)
        nx.draw(network, pos, cmap=cmap, node_color=node_colors, with_labels=w_labels, node_size=node_size)
        add_colorbar(node_colors)
        ax.set_title(title)
        plt.tight_layout()
    else:
        pos = nx.spring_layout(network)
        nx.draw(network, pos, cmap=cmap, with_labels=True, node_size=node_size)
        ax.set_title(title)
        plt.tight_layout()

    if figure_path is not '':
        plt.savefig(figure_path,
                    format='pdf',
                    bbox_inches='tight')


def investigate_d(network: nx.DiGraph, ds, colors, n_steps, d_figure_path):
    """
    Calculates PageRank at different values of the damping factor d and
    visualizes and saves results for interpretation

    Parameters
    ----------
    network : a NetworkX graph object
    ds : a list of d values
    colors : visualization color for PageRank at each d, must have same length as ds
    n_steps : int; number of steps taken in random walker algorithm
    d_figure_path : string
    """
    fig = plt.figure(1000)
    ax = fig.add_subplot(111)
    # Hint: use zip to loop over ds and colors at once
    for (d, color) in zip(ds, colors):
        pageRank_pi = page_rank(network, d, n_steps)
        nodes = sorted(network.nodes())
        node_colors = [pageRank_pi[node] for node in nodes]
        ax.plot(nodes, node_colors, color, label="$d=$" + str(d))

    ax.set_xlabel(r'Node index')
    ax.set_ylabel(r'PageRank')
    ax.set_title(r'PageRank with different damping factors')
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(d_figure_path, format='pdf', bbox_inches='tight')


def main():
    network = nx.DiGraph(nx.read_edgelist('pagerank_network.edg', create_using=nx.DiGraph()))

    # Visualization of the model network network (note that spring_layout
    # is intended to be used with undirected networks):
    node_positions = nx.layout.spring_layout(network.to_undirected())
    uncolored_figure_path = 'graphs/page_rank/uncolored_graph.pdf'
    visualize_network(network=network,
                      node_positions=node_positions,
                      figure_path=uncolored_figure_path,
                      title='network')
    nodes = network.nodes()
    n_nodes = len(nodes)

    # PageRank with self-written function
    n_steps = 10000
    d = .85
    pageRank_rw = page_rank(network, d, n_steps)

    # # Visualization of PageRank on network:
    node_colors = [pageRank_rw[node] for node in nodes]
    colored_figure_path = 'graphs/page_rank/colored_graph.pdf'
    visualize_network(network=network,
                      node_positions=node_positions,
                      figure_path=colored_figure_path,
                      node_colors=node_colors,
                      title='PageRank random walker')

    # PageRank with networkx:
    pageRank_nx = nx.pagerank(network)

    # Comparison of results from own function and nx.pagerank match:
    compare_pr(n_nodes, nodes, pageRank_nx, pageRank_rw)

    # PageRank with power iteration
    n_iterations = 10
    pageRank_pi = pagerank_poweriter(network, d, n_iterations)

    # Visualization of PageRank by power iteration on the network
    power_iteration_path = 'graphs/page_rank/power_iterations.pdf'
    node_colors = [pageRank_pi[node] for node in nodes]
    visualize_network(network,
                      node_positions,
                      power_iteration_path,
                      node_colors=node_colors,
                      title='PageRank power iteration')
    compare_pr(n_nodes, nodes, pageRank_nx, pageRank_pi)

    # Investigating the running time of the power iteration fuction
    running_time(network)


def running_time(network: nx.DiGraph):
    num_tests = 3
    n_nodes = 10 ** 4
    k5net = nx.directed_configuration_model(n_nodes * [5], n_nodes * [5], create_using=nx.DiGraph())

    # Print results: how many seconds were taken for the test network of 10**4 nodes,
    #  how many hours would a 26*10**6 nodes network take?
    result = timeit.timeit(lambda: pagerank_poweriter(k5net, 0.85, 10), number=num_tests)
    print(result)

    # Investigating the running time of the random walker function
    n_steps = 1000 * n_nodes  # each node gets visited on average 1000 times

    # Print results: how many seconds were taken for the test network of 10**4 nodes,
    # how many hours would a 26*10**6 nodes network take?
    result = timeit.timeit(lambda: page_rank(k5net, 0.85, n_steps), number=num_tests)
    print(result)

    # Investigating effects of d:
    ds = np.arange(0, 1.2, 0.2)
    colors = ['b', 'r', 'g', 'm', 'k', 'c']
    d_figure_path = 'graphs/page_rank/d_effect.pdf'
    investigate_d(network, ds, colors, n_steps, d_figure_path)

    # Wikipedia network:
    network_wp = nx.DiGraph(nx.read_edgelist("wikipedia_network.edg", create_using=nx.DiGraph()))

    # nx.read_edgelist.
    page_rank_wp = nx.pagerank(network_wp)
    indegree_wp = network_wp.in_degree()
    outdegree_wp = network_wp.out_degree()

    if page_rank_wp is not {}:
        highest_pr = sorted(page_rank_wp, key=lambda k: page_rank_wp[k])[::-1][0:5]
    print('---Highest PageRank:---')
    for p in highest_pr:
        print(page_rank_wp[p], ":", p)

    if indegree_wp is not {}:
        highest_id = sorted(indegree_wp, key=lambda k: indegree_wp[k])[::-1][0:5]
    print('---Highest In-degree:---')
    for p in highest_id:
        print(indegree_wp[p], ":", p)

    if outdegree_wp is not {}:
        highest_od = sorted(outdegree_wp, key=lambda k: outdegree_wp[k])[::-1][0:5]
    print('---Highest Out-degree:---')
    for p in highest_od:
        print(outdegree_wp[p], ":", p)


def compare_pr(n_nodes, nodes, pageRank_nx, pageRank_rw):
    pageRank_rw_array = np.zeros(n_nodes)
    pageRank_nx_array = np.zeros(n_nodes)

    for node in nodes:
        pageRank_rw_array[int(node)] = pageRank_rw[node]  # ordering dictionary values to node order
        pageRank_nx_array[int(node)] = pageRank_nx[node]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    check_figure_path = 'graphs/page_rank/check_sanity.pdf'

    # check visualization
    ax.plot(range(0, n_nodes), pageRank_rw_array, 'k+', label=r'power iteration')

    ax.plot(range(0, n_nodes), pageRank_nx_array, 'rx', label=r'networkx')

    ax.set_xlabel(r'Node index')
    ax.set_ylabel(r'PageRank')
    ax.set_title(r'PageRank with different methods')
    ax.legend(loc=0)

    plt.tight_layout()
    plt.savefig(check_figure_path, format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
