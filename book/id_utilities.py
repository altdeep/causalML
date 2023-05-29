import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import itertools
from IPython.display import Image
import pygraphviz as pgv

from y0.graph import NxMixedGraph
from y0.dsl import Probability
from y0.identify import is_identifiable

from typing import Tuple

def convert_graph(dag: nx.DiGraph, observational: list) -> Tuple[nx.DiGraph, nx.Graph]:
    """
    This function converts a nx.DiGraph DAG to the mixed graph (ADMG) format
    used by y0.  It removes unobserved nodes from the given DAG.  When B is
    removed from A -> B -> C, it creates a new edge A -> C.  When a common cause
    is removed, such as in A <- B -> C, it adds an undirected edge between A and
    C; these correspond to bidirected edges in the ADMG graph type A <-> C,
    which is how y0 algorithms reason about latent common cause variables.

    Warning: This code was created for pedagogical purposes and hasn't been
    well-tested.

    :param dag: The input directed acyclic graph.
    :param observational: List of nodes in the joint distribution of the data.
    :return: A tuple of two elements, first is the updated directed acyclic
             graph and second is the undirected graph created from the removed
             nodes.  Used to make an NxMixedGrpah.
    """
    observed_set = set(observational)  # for efficient lookups
    nodes_to_remove = [node for node in dag.nodes if node not in observed_set]
    undirected_graph = nx.Graph()
    directed_graph = dag.copy()

    for node in nodes_to_remove:
        predecessors = list(directed_graph.predecessors(node))
        successors = list(directed_graph.successors(node))

        # connect all parents with all children
        for parent in predecessors:
            for child in successors:
                directed_graph.add_edge(parent, child)
        
                # now remove the node
        directed_graph.remove_node(node)

        # add an edge between each pair of child nodes
        for child1, child2 in itertools.combinations(successors, 2):
            undirected_graph.add_edge(child1, child2)

    return directed_graph, undirected_graph


def check_identifiable(dag: NxMixedGraph, query: Probability, distribution: Probability):
    """
    This function checks if a probability query is identifiable given a DAG, and
    an observational distribution.  It converts the DAG to a mixed graph (ADMG)
    based what variables are latent in the observational distribution. This is
    the graph class that y0 uses to run identifiability algorithms.  It then
    runs y0's is_identifiable function.

    Warning: This code was created for pedagogical purposes and hasn't been
    well-tested.

    :param graph: The input mixed graph.
    :param query: The probability query to check.
    :param base_distribution: The base distribution.
    :return: Boolean value indicating whether the query is identifiable or not.
    """
    if len(dag.undirected.edges()) > 0:
        raise ValueError("This code is intended for DAGs.")
    observational_variables = distribution.get_variables()
    directed_graph, undirected_graph = convert_graph(
        dag.directed,
        observational=observational_variables)
    mixed_graph = NxMixedGraph.from_edges(
        directed=directed_graph.edges,
        undirected=undirected_graph.edges
    )
    return is_identifiable(mixed_graph, query)


def gv_draw(y0_graph: NxMixedGraph, observed: list = []):
    """
    Function to draw a y0 mixed graphs.  In the NxMixedGraph class,
    undirected edges correspond to missing common causes.
    This visualization replaces these edges with an explicit latent
    "N" variable.
    
    :param y0_graph: The input mixed graph.
    :param observed: List of nodes that are observed.
    :return: Image of the drawn graph.
    """
    def _format_node_labels(plot_graph):
        for node in plot_graph.nodes():
            if "N_" in str(node):
                _, var = node.split("_")
                new_label = f"<N<sup>{var}</sup>>"
                plot_graph.get_node(node).attr['label'] = new_label
                plot_graph.get_node(node).attr['main'] = var
            elif "@" in node:
                main, subscript = node.split(" @ ")
                subscript_var = subscript.strip("+").strip("-")
                subscript = subscript.lower()
                new_label = f"<{main}<sub>{subscript}</sub>>"
                plot_graph.get_node(node).attr['label'] = new_label
                plot_graph.get_node(node).attr['main'] = main
                plot_graph.get_node(node).attr['subscript'] = subscript
                plot_graph.get_node(node).attr['subscript_var'] = subscript_var
            else:
                plot_graph.get_node(node).attr['main'] = str(node)
            
    def _set_node_color(plot_graph, observed):
        for node_name in plot_graph.nodes():
            if node_name not in observed:
                node = plot_graph.get_node(node_name)
                node.attr['fillcolor'] = "lightgray"
                node.attr['style'] = 'filled'
        if observed is None:
            observed = []
        
    plot_graph = pgv.AGraph(strict=False, directed=True)
    directed = y0_graph.directed
    directed_graph = to_agraph(directed)
    for u, v in directed_graph.edges():
        plot_graph.add_edge(u, v)
    nx_common_cause = y0_graph.undirected
    has_latents = len(nx_common_cause.edges) > 0
    common_cause_graph = to_agraph(nx_common_cause)
    node_to_noise = {}
    main_to_noise = {}
    bad_edges = []
    non_noise = directed_graph.nodes()
    intervention_targets = set()
    
    # Adding dashed edges and setting node labels
    for u, v in common_cause_graph.edges():
        main_var = u
        if "@" in u:
            main_var, _ = u.split(" @ ")
        noise_term = f"N_{main_var}"
        main_to_noise[main_var] = noise_term
        if not plot_graph.has_edge(noise_term, u):
            plot_graph.add_edge(noise_term, u, style="dashed")
            node_to_noise[u] = plot_graph.get_node(noise_term)
        if not plot_graph.has_edge(noise_term, v):
            plot_graph.add_edge(noise_term, v, style="dashed")
            node_to_noise[v] = plot_graph.get_node(noise_term)

    _format_node_labels(plot_graph)
            
    # Modify nodes
    if has_latents:
        for node in non_noise:
            if node not in node_to_noise.keys():
                intervention_targets.add(node)
                node = plot_graph.get_node(node)
                node.attr['shape'] = "rectangle"
                main = node.attr['main']
                subscript = node.attr['subscript']
                label = f'do({main}={subscript})'
                node.attr['label'] = label
                noise_term = main_to_noise[main]
                bad_edges.append((noise_term, node))
                plot_graph.add_edge(noise_term, node)

    plot_graph.layout(prog='dot')
    for edge in bad_edges:
        plot_graph.remove_edge(edge)
    
    # Color latents gray.
    if len(observed) > 0:
        _set_node_color(plot_graph, observed + [str(n) for n in intervention_targets])
    else:
        _set_node_color(plot_graph, non_noise)
    
    plot_graph.draw('plot_graph.png')
    return Image('plot_graph.png')
