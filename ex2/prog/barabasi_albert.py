from random import randint, choice, seed
from typing import List, Set

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl

from prog.utils import graph_directory


def select_random_set(seq: List, m: int) -> Set:
    targets = set()
    while len(targets) < m:
        targets.add(np.random.choice(seq))
    return targets


def barabasi_albert_network_generator(n: int, m: int) -> nx.Graph:
    assert n > 1 and m > 0

    graph = nx.complete_graph(m + 1)
    targets = list(range(m + 1))
    degrees_list = []

    start = m + 2
    for curr_node in range(start, n):
        if curr_node != start:
            edges = zip([curr_node] * m, targets)
            graph.add_edges_from(edges)

        degrees_list.extend(targets)
        degrees_list.extend([curr_node] * m)
        targets = select_random_set(degrees_list, m)

    return graph


def question_a():
    graph = barabasi_albert_network_generator(200, 1)
    nx.draw_spring(graph)
    plt.savefig(graph_directory + "barabasi_albert.pdf")


def probability_degrees_distribution(m, k):
    return (2 * m * (m + 1)) / (k * (k + 1) * (k + 2))


def question_b():
    graph = barabasi_albert_network_generator(10 ** 4, 2)
    degrees = list(nx.degree(graph).values())

    # binning tutorial
    bins = [np.min(degrees)]
    cur_value = bins[0]
    mult = 2.5
    while cur_value < np.max(degrees):
        cur_value *= mult
        bins.append(cur_value)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog()
    ax.set_title("degrees distribution")
    ax.set_xlabel('degrees')
    ax.set_ylabel('probability')

    ax.hist(degrees, normed=True, bins=bins, log=True, label="Experimental degree distribution")

    x = list(range(np.min(degrees), int(np.max(bins[-1]))))
    y = [probability_degrees_distribution(2, k) for k in x]
    ax.plot(x, y, 'r--', label="Therorical degree distribution")
    ax.legend(loc="best")
    fig.savefig(graph_directory + "degree_distribution.pdf")


def barabasi_albert():
    question_a()
    question_b()


if __name__ == '__main__':
    barabasi_albert()
