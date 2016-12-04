import networkx as nx

graph_directory = "graphs/"


def average_degree(graph: nx.Graph):
    return (2 * len(graph.edges())) / len(graph)
