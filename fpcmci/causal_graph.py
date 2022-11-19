from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from netgraph import Graph
from fpcmci.utilities.constants import *


def dag(res,
        node_layout = 'dot',
        min_width = 1,
        max_width = 5,
        min_score = 0,
        max_score = 1,
        node_size = 8,
        node_color = 'orange',
        edge_color = 'grey',
        font_size = 12,
        show_edge_labels = True,
        label_type = LabelType.Lag,
        save_name = None):
    """
    build a dag

    Args:
        result (dict): dependencies result
        node_layout (str, optional): Node layout. Defaults to 'dot'.
        min_width (int, optional): minimum linewidth. Defaults to 1.
        max_width (int, optional): maximum linewidth. Defaults to 5.
        min_score (int, optional): minimum score range. Defaults to 0.
        max_score (int, optional): maximum score range. Defaults to 1.
        node_size (int, optional): node size. Defaults to 8.
        node_color (str, optional): node color. Defaults to 'orange'.
        edge_color (str, optional): edge color. Defaults to 'grey'.
        font_size (int, optional): font size. Defaults to 12.
        show_edge_labels (bool, optional): bit to show the label of the dependency on the edge/node border. Defaults to True.
        label_type (Enum, optional): LAG/SCORE information of the dependency on the edge/node border. Defaults to LAG.
        save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
    """

    G = nx.DiGraph()

    # add nodes
    G.add_nodes_from(res.keys())
    
    border = dict()
    for t in res.keys():
        border[t] = 0
        for s in res[t]:
            if t == s[SOURCE]:
                border[t] = __scale(s[SCORE], min_width, max_width, min_score, max_score)
    
    if show_edge_labels:
        if label_type == LabelType.Lag:
            node_label = {t: s[LAG] for t in res.keys() for s in res[t] if t == s[SOURCE]}
        elif label_type == LabelType.Score:
            node_label = {t: round(s[SCORE], 3) for t in res.keys() for s in res[t] if t == s[SOURCE]}
    else:
        node_label = None

    # edges definition
    edges = [(s[SOURCE], t) for t in res.keys() for s in res[t] if t != s[SOURCE]]
    G.add_edges_from(edges)
    
    edge_width = {(s[SOURCE], t): __scale(s[SCORE], min_width, max_width, min_score, max_score) for t in res.keys() for s in res[t] if t != s[SOURCE]}
    if show_edge_labels:
        if label_type == LabelType.Lag:
            edge_label = {(s[SOURCE], t): s[LAG] for t in res.keys() for s in res[t] if t != s[SOURCE]}
        elif label_type == LabelType.Score:
            edge_label = {(s[SOURCE], t): round(s[SCORE], 3) for t in res.keys() for s in res[t] if t != s[SOURCE]}
    else:
        edge_label = None

    fig, ax = plt.subplots(figsize=(8,6))

    if edges:
        a = Graph(G, 
                node_layout = node_layout,
                node_size = node_size,
                node_color = node_color,
                node_labels = node_label,
                node_edge_width = border,
                node_label_fontdict = dict(size=font_size),
                node_edge_color = edge_color,
                node_label_offset = 0.1,
                node_alpha = 1,
                
                arrows = True,
                edge_layout = 'curved',
                edge_label = show_edge_labels,
                edge_labels = edge_label,
                edge_label_fontdict = dict(size=font_size),
                edge_color = edge_color, 
                edge_width = edge_width,
                edge_alpha = 1,
                edge_zorder = 1,
                edge_label_position = 0.35,
                edge_layout_kwargs = dict(bundle_parallel_edges = False, k = 0.05))
        
        nx.draw_networkx_labels(G, 
                                pos = a.node_positions,
                                labels = {n: n for n in G},
                                font_size = font_size)

    if save_name is not None:
        plt.savefig(save_name, dpi = 300)
    else:
        plt.show()
               
        
def ts_dag(res,
           tau,
           min_width = 1,
           max_width = 5,
           min_score = 0,
           max_score = 1,
           node_size = 8,
           node_color = 'orange',
           edge_color = 'grey',
           font_size = 12,
           save_name = None):
    """
    build a timeseries dag

    Args:
        result (dict): dependencies result
        tau (int): max time lag
        min_width (int, optional): minimum linewidth. Defaults to 1.
        max_width (int, optional): maximum linewidth. Defaults to 5.
        min_score (int, optional): minimum score range. Defaults to 0.
        max_score (int, optional): maximum score range. Defaults to 1.
        node_size (int, optional): node size. Defaults to 8.
        node_color (str, optional): node color. Defaults to 'orange'.
        edge_color (str, optional): edge color. Defaults to 'grey'.
        font_size (int, optional): font size. Defaults to 12.
        save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
    """
    
    # add nodes
    G = nx.grid_2d_graph(tau + 1, len(res.keys()))
    pos = dict()
    for n in G.nodes():
        if n[0] == 0:
            pos[n] = (n[0], n[1]/2)
        else:
            pos[n] = (n[0] + .5, n[1]/2)
    scale = max(pos.values())
    G.remove_edges_from(G.edges())

    # edges definition
    edges = list()
    edge_width = dict()
    for t in res.keys():
        for s in res[t]:
            s_index = len(res.keys())-1 - list(res.keys()).index(s[SOURCE])
            t_index = len(res.keys())-1 - list(res.keys()).index(t)
            s_node = (tau - s[LAG], s_index)
            t_node = (tau, t_index)
            edges.append((s_node, t_node))
            edge_width[(s_node, t_node)] = __scale(s[SCORE], min_width, max_width, min_score, max_score)
    G.add_edges_from(edges)

    # label definition
    labeldict = {}
    for n in G.nodes():
        if n[0] == 0:
            labeldict[n] = list(res.keys())[len(res.keys()) - 1 - n[1]]

    fig, ax = plt.subplots(figsize=(8,6))

    # time line text drawing
    pos_tau = set([pos[p][0] for p in pos])
    max_y = max([pos[p][1] for p in pos])
    for p in pos_tau:
        if abs(int(p) - tau) == 0:
            ax.text(p, max_y + .3, r"$t$", horizontalalignment='center', fontsize=font_size)
        else:
            ax.text(p, max_y + .3, r"$t-" + str(abs(int(p) - tau)) + "$", horizontalalignment='center', fontsize=font_size)

    Graph(G,
          node_layout = {p : np.array(pos[p]) for p in pos},
          node_size = node_size,
          node_color = node_color,
          node_labels = labeldict,
          node_label_offset = 0,
          node_edge_width = 0,
          node_label_fontdict = dict(size=font_size),
          node_alpha = 1,
          
          arrows = True,
          edge_layout = 'curved',
          edge_label = False,
          edge_color = edge_color, 
          edge_width = edge_width,
          edge_alpha = 1,
          edge_zorder = 1,
          scale = (scale[0] + 2, scale[1] + 2))

    if save_name is not None:
        plt.savefig(save_name, dpi = 300)
    else:
        plt.show()


def __scale(score, min_width, max_width, min_score = 0, max_score = 1):
    """
    Scales the score of the cause-effect relationship strength to a linewitdth

    Args:
        score (float): score to scale
        min_width (float): minimum linewidth
        max_width (float): maximum linewidth
        min_score (int, optional): minimum score range. Defaults to 0.
        max_score (int, optional): maximum score range. Defaults to 1.

    Returns:
        float: scaled score
    """
    return ((score - min_score) / (max_score - min_score)) * (max_width - min_width) + min_width
