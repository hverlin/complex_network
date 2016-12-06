import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import operator


def plot_largest_component(x, y_values, labels, styles, xlabel, ylabel, title, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for (y_value, label, style) in zip(y_values, labels, styles):
        ax.plot(x, y_value, style, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")

    fig.savefig(path)


def remove_edges(edges, network):
    val = []
    for (u, v) in edges:
        network.remove_edge(u, v)
        largest_cc = max(nx.connected_components(network), key=len)
        val.append(len(largest_cc))
    return val


def att_descending_link_weight(network: nx.Graph):
    edges = ((u, v) for (u, v, _) in reversed(sorted(network.edges(data=True), key=lambda tup: tup[2]["weight"])))
    return remove_edges(edges, network)


def att_ascending_link_weight(network: nx.Graph):
    edges = ((u, v) for (u, v, _) in sorted(network.edges(data=True), key=lambda tup: tup[2]["weight"]))
    return remove_edges(edges, network)


def att_random_order(network: nx.Graph):
    edges = list(network.edges())
    np.random.shuffle(edges)
    return remove_edges(edges, network)


def att_decending_betweenness_centrality(network: nx.Graph):
    bc = nx.edge_betweenness_centrality(network, weight="weight")
    edges = (e for (e, _) in reversed(sorted(bc.items(), key=operator.itemgetter(1))))
    return remove_edges(edges, network)


def main():
    network = nx.read_weighted_edgelist('OClinks_w_undir.edg')

    labels = [
        "descending link weight",
        "ascending link weight",
        "random order",
        "descending order of edge betweenness centrality"
    ]

    styles = ["-", "--", "-.", ":"]

    y_values = [
        att_descending_link_weight(network.copy()),
        att_ascending_link_weight(network.copy()),
        att_random_order(network.copy()),
        att_decending_betweenness_centrality(network.copy())
    ]

    x_values = np.divide(list(range(len(list(network.edges())))), len(list(network.edges())))

    plot_largest_component(
        x=x_values,
        y_values=y_values,
        styles=styles,
        labels=labels, xlabel="fraction of removed links",
        ylabel="largest component size",
        title="attack tolerance of network against different attack schemes",
        path="graphs/ex2/attacks.pdf"
    )


if __name__ == '__main__':
    main()
