import networkx as nx
import matplotlib.pyplot as plt
# Your dictionary of lists
a = {1: [2, 3], 2: [4], 3: [1], 4: [2, 1]}

# Create a graph from the dictionary
G = nx.from_dict_of_lists(a, create_using=nx.DiGraph)

# You can specify create_using=nx.Graph if you want an undirected graph

# Print the nodes and edges in the graph
print("Nodes in the graph:", G.nodes())
print("Edges in the graph:", G.edges())

layout = nx.spring_layout(G)

# Draw the nodes and edges
nx.draw(G, pos=layout, with_labels=True, node_color='skyblue', font_size=10, node_size=500)
plt.title("NetworkX Graph Visualization")
plt.show()