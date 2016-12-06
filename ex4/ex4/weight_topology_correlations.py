# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from scipy.stats import binned_statistic


def get_link_weights(net: nx.Graph):
    """
    Returns a list of link weights in the network.
    
    Parameters
    -----------
    net: a networkx.Graph() object
    
    Returns
    --------
    weights: list of link weights in net
    """
    # Hints:
    # to get the links with their weight data, use net.edges(data=True)
    # to get weight of a single link, use (i, j, data) = link, 
    # weight = data['weight']
    weights = []
    for (i, j, data) in net.edges(data=True):
        weights.append(data["weight"])
    return weights


def ccdf(data, ax, label, marker):
    """
    Plot the complementary cumulative distribution function
    (1-CDF(x)) based on the data on the axes object.
    """
    sorted_vals = np.sort(np.unique(data))
    ccdf = np.zeros(len(sorted_vals))
    n = float(len(data))
    for i, val in enumerate(sorted_vals):
        ccdf[i] = np.sum(data >= val) / n
    ax.plot(sorted_vals, ccdf, marker, label=label)


def plot_ccdf(datavecs, labels, styles, xlabel, ylabel, num, path):
    """
    Plots in a single figure the complementary cumulative distributions (1-CDFs)
    of the given data vectors.
    
    Parameters
    -----------
    datavecs: data vectors to plot, a list of iterables
    labels: labels for the data vectors, list of strings
    styles = styles in which plot the distributions, list of strings
    xlabel: x label for the figure, string
    ylabel: y label for the figure, string
    num: an id of the figure, int or string
    path: path where to save the figure, string
    """
    fig = plt.figure(num)
    ax = fig.add_subplot(111)
    for datavec, label, style in zip(datavecs, labels, styles):
        ax.loglog()
        ccdf(data=datavec, ax=ax, label=label, marker=style)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=0)
    ax.grid()
    fig.savefig(path)
    print('1-CDF figure saved to ' + path)


def create_linbins(start, end, n_bins):
    """
    Creates a set of linear bins.
    
    Parameters
    -----------
    start: minimum value of data, int
    end: maximum value of data, int
    n_bins: number of bins, int
    
    Returns
    --------
    bins: a list of linear bin edges
    """
    bins = np.linspace(start, end, n_bins)
    return bins


def create_logbins(start, end, n_log, n_lin=0):
    """
    Creates a combination of linear and logarithmic bins: n_lin linear bins 
    of width 1 starting from start and n_log logarithmic bins further to
    max.
    
    Parameters
    -----------
    start: starting point for linear bins, float
    end: maximum value of data, int
    n_log: number of logarithmic bins, int
    n_lin: number of linear bins, int
    
    Returns
    -------
    bins: a list of bin edges
    """
    bins = None
    if n_lin == 0:
        bins = np.logspace(np.log10(start), np.log10(end), n_log)
    elif n_lin > 0:
        bins = np.array(
            [start + i for i in range(n_lin)] + list(np.logspace(np.log10(start + n_lin), np.log10(end), n_log)))
    return bins


def get_link_overlap(net: nx.Graph):
    """
    Calculates link overlap: 
    O_ij = n_ij / [(k_i - 1) + (k_j - 1) + 1]
    
    Parameters
    -----------
    net: a networkx.Graph() object
    
    Returns
    --------
    overlaps: list of link overlaps in net
    """
    # Hint: for getting common neighbors of two nodes, use
    # set datatype and intersection method
    overlaps = []
    for (u, v) in net.edges():
        u_neighbors = set(net.neighbors(u))
        v_neighbors = set(net.neighbors(v))
        n_uv = len(u_neighbors.intersection(v_neighbors))
        overlaps.append(n_uv / ((net.degree(u) - 1) + (net.degree(v) - 1) + 1))
    return overlaps


def main():
    network = nx.read_weighted_edgelist('OClinks_w_undir.edg')
    net_name = 'fb_like'

    #  compute CCDF plot
    degree_vec, strength_vec, weights = get_degree_strength_weights(network)
    compute_ccdf(degree_vec, net_name, strength_vec, weights)

    # calculating average link weight per node
    strengths = dict(nx.degree(network, weight="weight"))
    av_weight = [strengths[node] / network.degree(node) for node in network]

    # creating scatters and adding bin averages on top of them
    min_deg = min(degree_vec)
    max_deg = max(degree_vec)
    n_bins = 50

    linbins = create_linbins(min_deg, max_deg, n_bins)
    logbins = create_logbins(0.5, max_deg, n_bins, n_lin=10)
    num = 'b) ' + net_name + '_'
    base_path = 'graphs/ex1/average_link'  # A scale-related suffix will be added to this base path
    alpha = 0.1  # transparency of data points in the scatter

    for bins, scale in zip([linbins, logbins], ['linear', 'log']):
        fig = plt.figure(num + scale)
        ax = fig.add_subplot(111)

        # mean degree value of each degree bin
        degree_bin_means, _, _ = binned_statistic(degree_vec, values=degree_vec, bins=bins, statistic='mean')

        # mean strength value of each degree bin
        strength_bin_means, _, _ = binned_statistic(degree_vec, values=strength_vec, bins=bins, statistic='mean')

        # number of points in each degree bin
        counts, _, _ = binned_statistic(degree_vec, values=degree_vec, bins=bins, statistic='count')

        # plotting all points
        ax.scatter(degree_vec, av_weight, marker='o', s=1.5, alpha=alpha, label="unbinned data")

        # plotting bin average, marker size scaled by number of data points in the bin
        bin_av_weight = np.divide(strength_bin_means, degree_bin_means)
        ax.scatter(degree_bin_means,
                   bin_av_weight,
                   marker='o',
                   color='g',
                   s=np.sqrt(counts) + 1,
                   label='binned data')
        ax.set_xscale(scale)

        min_max = np.array([min_deg, max_deg])
        ax.set_xlabel(r'degree $k$')
        ax.set_ylabel(r'avg. link weight $s$')
        ax.grid()

        ax.legend(loc='best')
        plt.suptitle('avg. link weight vs. strength:' + net_name)
        save_path = base_path + scale + '_' + net_name + '.pdf'
        fig.savefig(save_path)
        print('Average link weight scatter saved to ' + save_path)

    # getting link neighborhood overlaps
    overlaps = get_link_overlap(network)

    # creating link neighborhood overlap scatter
    num = 'd) + net_name'
    fig = plt.figure(num)
    ax = fig.add_subplot(111)

    n_bins = 50
    min_w = np.min(weights)
    max_w = np.max(weights)

    linbins = create_linbins(min_w, max_w, n_bins)
    logbins = create_logbins(min_w, max_w, n_bins)

    bins = logbins

    # mean weight value of each weight bin
    weight_bin_means, _, _ = binned_statistic(weights, values=weights, bins=bins, statistic='mean')

    # mean link neighborhood overlap of each weight bin
    overlap_bin_means, _, _ = binned_statistic(weights, values=overlaps, bins=bins, statistic='mean')

    # number of points in each weigth bin
    counts, _, _ = binned_statistic(weights, values=overlaps, bins=bins, statistic='count')

    # plotting all points
    ax.scatter(weights, overlaps, marker="o", s=1.5, alpha=alpha, label="unbinned data")

    # plotting bin average, marker size scaled by number of data points in the bin
    ax.scatter(
        weight_bin_means,
        overlap_bin_means,
        s=np.sqrt(counts) + 2,
        marker='o',
        color='g',
        label="unbinned data"
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel('Link weight')
    ax.set_ylabel('Link neighborhood overlap')
    fig.suptitle('Overlap vs. weight:' + net_name)
    save_path = 'graphs/ex1/overlap.pdf'
    fig.savefig(save_path)
    print('Link neighborhood overlap scatter saved as ' + save_path)


def get_degree_strength_weights(network):
    degrees = dict(nx.degree(network))
    strengths = dict(nx.degree(network, weight="weight"))
    degree_vec, strength_vec = [], []
    for node in network.nodes():
        degree_vec.append(degrees[node])
        strength_vec.append(strengths[node])
    weights = get_link_weights(network)
    return degree_vec, strength_vec, weights


def compute_ccdf(degree_vec, net_name, strength_vec, weights):
    datavecs = [degree_vec, strength_vec, weights]
    labels = ['degree', 'strength', 'weigth']
    styles = ['-', '--', '-.']
    xlabel = 'Degree, strength or link weight'
    ylabel = '1-CDF'
    num = 'a)' + net_name + ".pdf"
    path = 'graphs/ex1/' + num
    plot_ccdf(datavecs, labels, styles, xlabel, ylabel, num, path)


if __name__ == '__main__':
    main()
