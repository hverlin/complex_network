import networkx as nx
import pylab as pl

from prog.utils import average_degree, graph_directory


def analytics_averages(p):
    return 2 * p, p ** 3, -2 * p ** 3 + 3 * p


def get_averages(n: int, p: float, nb: int = 100):
    i = 0
    while i < (nb):
        graph = nx.fast_gnp_random_graph(n, p)
        yield \
            average_degree(graph), \
            nx.average_clustering(graph), \
            max(nx.diameter(g) for g in nx.connected_component_subgraphs(graph))
        i += 1


def plot_averages(file_name, n, nb_iterations=10):
    p_values, deg_values, c_values, d_values = [], [], [], []

    deg_ana, clus_ana, d_ana = [], [], []
    if n != 3:
        deg_ana, clus_ana, d_ana = None, None, None

    for p in pl.frange(0.00, 1, 0.05):
        p_values.append(p)

        if n == 3:
            degree, clustering, shortest_path = analytics_averages(p)
            deg_ana.append(degree)
            clus_ana.append(clustering)
            d_ana.append(shortest_path)

        degree, clustering, shortest_path = 0, 0, 0
        for (deg, clus, sp) in get_averages(n, p, nb_iterations):
            degree += deg
            clustering += clus
            shortest_path += sp

        deg_values.append(degree / nb_iterations)
        c_values.append(clustering / nb_iterations)
        d_values.append(shortest_path / nb_iterations)

    plot_average(c_values, file_name + "avg_cls.pdf", p_values,
                 "average clustering coefficient",
                 "Average clustering coefficient against p",
                 clus_ana)

    plot_average(d_values, file_name + "avg_d.pdf", p_values,
                 "Average diameter",
                 "Average diameter against p",
                 d_ana)

    plot_average(deg_values, file_name + "avg_deg.pdf", p_values,
                 "Average degree",
                 "Average degree against p",
                 deg_ana)


def plot_average(values, file_name, p_values, ylabel, title, ana_values=None):
    fig = pl.plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=p_values, y=values, color="b")

    if ana_values is not None:
        ax.plot(p_values, ana_values, color="r")

    ax.set_xlabel("probability of a link")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(file_name)


def question_e():
    plot_averages(graph_directory + "_N=3_", 3)
    plot_averages(graph_directory + "_N=100_", 100)


if __name__ == '__main__':
    question_e()
