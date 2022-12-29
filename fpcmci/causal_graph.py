from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from netgraph import Graph
from fpcmci.basics.constants import *


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
        label_type = LabelType.Lag,
        save_name = None):
    """
    build a dag

    Args:
        res (dict): dependencies result
        node_layout (str, optional): Node layout. Defaults to 'dot'.
        min_width (int, optional): minimum linewidth. Defaults to 1.
        max_width (int, optional): maximum linewidth. Defaults to 5.
        min_score (int, optional): minimum score range. Defaults to 0.
        max_score (int, optional): maximum score range. Defaults to 1.
        node_size (int, optional): node size. Defaults to 8.
        node_color (str, optional): node color. Defaults to 'orange'.
        edge_color (str, optional): edge color. Defaults to 'grey'.
        font_size (int, optional): font size. Defaults to 12.
        label_type (LabelType, optional): enum to set whether to show the lag time (LabelType.Lag) or the strength (LabelType.Score) of the dependencies on each link/node or not showing the labels (LabelType.NoLabels). Default LabelType.Lag.
        save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
    """

    G = nx.DiGraph()

    # NODES DEFINITION
    G.add_nodes_from(res.keys())
    
    # BORDER LINE
    border = dict()
    for t in res.keys():
        border[t] = 0
        for s in res[t]:
            if t == s[SOURCE]:
                border[t] = max(__scale(s[SCORE], min_width, max_width, min_score, max_score), border[t])
    
    # BORDER LABEL
    node_label = None
    if label_type == LabelType.Lag or label_type == LabelType.Score:
        node_label = {t: [] for t in res.keys()}
        for t in res.keys():
            for s in res[t]:
                if t == s[SOURCE]:
                    if label_type == LabelType.Lag:
                        node_label[t].append(s[LAG])
                    elif label_type == LabelType.Score:
                        node_label[t].append(round(s[SCORE], 3))
            node_label[t] = ",".join(str(s) for s in node_label[t])


    # EDGE DEFINITION
    edges = [(s[SOURCE], t) for t in res.keys() for s in res[t] if t != s[SOURCE]]
    G.add_edges_from(edges)
    
    # EDGE LINE
    edge_width = {(s[SOURCE], t): 0 for t in res.keys() for s in res[t] if t != s[SOURCE]}
    for t in res.keys():
        for s in res[t]:
            if t != s[SOURCE]:
                edge_width[(s[SOURCE], t)] = max(__scale(s[SCORE], min_width, max_width, min_score, max_score), edge_width[(s[SOURCE], t)])
    
    # EDGE LABEL
    edge_label = None
    if label_type == LabelType.Lag or label_type == LabelType.Score:
        edge_label = {(s[SOURCE], t): [] for t in res.keys() for s in res[t] if t != s[SOURCE]}
        for t in res.keys():
            for s in res[t]:
                if t != s[SOURCE]:
                    if label_type == LabelType.Lag:
                        edge_label[(s[SOURCE], t)].append(s[LAG])
                    elif label_type == LabelType.Score:
                        edge_label[(s[SOURCE], t)].append(round(s[SCORE], 3))
        for k in edge_label.keys():
            edge_label[k] = ",".join(str(s) for s in edge_label[k])

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
                edge_label = label_type != LabelType.NoLabels,
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
        res (dict): dependencies result
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
        (float): scaled score
    """
    return ((score - min_score) / (max_score - min_score)) * (max_width - min_width) + min_width
