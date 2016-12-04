from random import randint
from random import seed
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from prog.utils import graph_directory


def draw_graph(graph, n, p):
    nx.draw_circular(graph)
    plt.savefig(graph_directory + 'watts_strogatz_' + str(n) + '_' + str(p) + '.pdf')
    plt.clf()


def generate_watts_strogatz_graph(n: int, m: int, p: float) -> nx.Graph:
    seed()

    graph = nx.Graph()
    graph.add_nodes_from(range(0, n))

    for node in graph.nodes_iter():
        for i in range(1, m + 1):
            graph.add_edge(node, np.remainder(node - i, n))
            graph.add_edge(node, np.remainder(node + i, n))

    edges = graph.edges()
    for (u, v) in edges:
        if np.random.rand() < p:
            graph.remove_edge(u, v)
            graph.add_edge(v, randint(0, n - 1))

    return graph


def frange_mul(mul: float, n: float):
    p = 0.001
    while p <= n:
        yield p
        p *= mul


def question_a():
    for (n, m, p) in [(15, 2, 0.0), (15, 2, 0.5)]:
        draw_graph(generate_watts_strogatz_graph(n, m, p), n, p)


def question_b():
    N = 1000
    m = 5

    avg_c_zero = nx.average_clustering(generate_watts_strogatz_graph(N, m, 0))
    avg_l_zero = nx.average_shortest_path_length(generate_watts_strogatz_graph(N, m, 0))

    p_values, ravg_c_values, ravg_l_values = [], [], []
    for p in frange_mul(2, 0.512):
        g = generate_watts_strogatz_graph(N, m, p)

        try:
            p_values.append(p)
            ravg_c_values.append(nx.average_clustering(g) / avg_c_zero)
            ravg_l_values.append(nx.average_shortest_path_length(g) / avg_l_zero)

        except Exception as e:
            print(e)
            print("err for p={p}".format(p=p))

    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)

    ax.semilogx()

    ax.scatter(x=p_values, y=ravg_c_values, color="b", label='relative average clustering')
    ax.scatter(x=p_values, y=ravg_l_values, color="r", label='relative average shortest path')

    ax.set_xlabel("probability of a link")
    ax.set_ylabel("average values")
    ax.set_title('Averages against p')
    ax.legend(loc='best')

    fig.savefig(graph_directory + "fig4.pdf")


def watts_strogatz():
    question_a()
    question_b()


if __name__ == '__main__':
    watts_strogatz()
