import gmsh
import logging
from amworkflow.src.constants.exceptions import GmshUseBeforeInitializedException
import networkx as nx
import matplotlib.pyplot as plt

def mesh_visualizer():
    try:
        gmsh.is_initialized()
    except:
        raise GmshUseBeforeInitializedException()
    gmsh.fltk.run()
    
def color_background(value):
    if value:
        return 'background-color: green'
    # else:
    #     return 'background-color: blue'
    
def plot_digraph(dataset: dict) -> None:
    G = nx.from_dict_of_lists(dataset, create_using=nx.DiGraph)

    layout = nx.spring_layout(G)

    # Draw the nodes and edges
    nx.draw(G, pos=layout, with_labels=True, node_color='skyblue', font_size=10, node_size=500)
    plt.title("NetworkX Graph Visualization")
    plt.show()