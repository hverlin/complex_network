import networkx as nx
import pylab as pl
from matplotlib import pyplot as plt

from prog.utils import graph_directory, average_degree


def generate_graph(n: int, p: float):
    graph = nx.fast_gnp_random_graph(n, p)
    k = average_degree(graph=graph)
    largest_components = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)][0:2]

    if len(largest_components) < 2:
        largest_components.append(0)

    return k, largest_components


def graphs_generator(start, stop, step, n):
    return (generate_graph(n, k / (n - 1)) for k in pl.frange(start, stop, step))


def question_d():
    n = 10 ** 4

    deg, lc1, lc2 = [], [], []

    for (k, lc) in graphs_generator(0.00, 2.5, 0.005, n=n):
        deg.append(k)
        lc1.append(lc[0])
        lc2.append(lc[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x=deg, y=lc1, color="b", label='largest component', marker='.')
    ax.scatter(x=deg, y=lc2, color="r", label='2nd largest component', marker='.')

    ax.set_xlabel("degrees")
    ax.set_ylabel("component size")
    ax.set_title('Size of the largest and the second largest\nconnected components against <k>')
    ax.legend(loc="center right")
    fig.savefig(graph_directory + "fig1.pdf")

    fig.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=deg, y=lc2, color="r", label='2nd largest component', marker='.')
    ax.set_xlabel("degrees")
    ax.set_ylabel("component size")
    ax.set_title('Size of the  second largest\nconnected component against <k>')
    fig.savefig(graph_directory + "fig1_2.pdf")


if __name__ == '__main__':
    question_d()
