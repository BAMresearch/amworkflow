import gmsh
import logging
from amworkflow.src.constants.exceptions import GmshUseBeforeInitializedException
import networkx as nx
import matplotlib.pyplot as plt
from amworkflow.src.geometries.property import shortest_distance_line_line
import numpy as np

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
    
def plot_intersect(x11 = 0, x12 = 0, y11 = 0, y12 = 0, x21 = 0, x22 = 0, y21 = 0, y22 = 0,segment1 = None, segment2 = None):
    if (segment1 is None) and (segment2 is None):
        # Coordinates for the two segments
        segment1_x = [x11, x12]
        segment1_y = [y11, y12]

        segment2_x = [x21, x22]
        segment2_y = [y21, y22]
        intersect = shortest_distance_line_line(np.array([[x11, y11,0], [x12, y12,0]]), np.array([[x21, y21,0], [x22, y22,0]]))[1]
        print(shortest_distance_line_line(np.array([[x11, y11,0], [x12, y12,0]]), np.array([[x21, y21,0], [x22, y22,0]]))[0])
    else:
        segment1_x = segment1.T[0]
        segment1_y = segment1.T[1]
        segment2_x = segment2.T[0]
        segment2_y = segment2.T[1]
        intersect = shortest_distance_line_line(segment1, segment2)[1]


    
    # Coordinates for the single point
    
    point_x_1 = [intersect[0].T[0]]
    point_y_1 = [intersect[0].T[1]]
    point_x_2 = [intersect[1].T[0]]
    point_y_2 = [intersect[1].T[1]]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the two segments
    ax.plot(segment1_x, segment1_y, color='blue', linestyle='-', linewidth=2, label='Segment 1')
    ax.plot(segment2_x, segment2_y, color='green', linestyle='-', linewidth=2, label='Segment 2')

    # Plot the single point
    ax.plot(point_x_1, point_y_1, marker='o', markersize=8, color='red', label='Point1')
    ax.plot(point_x_2, point_y_2, marker='o', markersize=8, color='red', label='Point2')

    # Add labels for the point and segments
    ax.text(2, 3, f'Point ({intersect[0]}, {intersect[1]})', fontsize=12, ha='right')
    ax.text(1, 2, 'Segment 1', fontsize=12, ha='right')
    ax.text(6, 3, 'Segment 2', fontsize=12, ha='right')

    # Add a legend
    ax.legend()

    # Set axis limits for better visualization
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)

    # Set axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Set plot title
    ax.set_title('Two Segments and One Point')
    plt.tight_layout()
    # Display the plot
    plt.show()