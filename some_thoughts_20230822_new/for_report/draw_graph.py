import os
import networkx as nx
import ast
from pyvis.network import Network

def extract_functions_classes(file_path):
    functions = {}
    classes = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                functions[function_name] = os.path.basename(file_path)  # Use script file name
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                classes[class_name] = os.path.basename(file_path)  # Use script file name
    
    return functions, classes

def find_function_calls(file_path, functions):
    calls = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                function_name = node.func.id
                if function_name in functions:
                    calls[function_name] = os.path.basename(file_path)  # Use script file name
    
    return calls

def build_dependency_graph(directory):
    G = nx.DiGraph()
    functions = {}
    classes = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_functions, file_classes = extract_functions_classes(file_path)
                functions.update(file_functions)
                classes.update(file_classes)
                folder_name = os.path.basename(root)
                G.add_node(folder_name, color='lightblue')  # Use a light shade of blue for folders
                G.add_edge(folder_name, file, color='blue')  # Use a slightly darker shade of blue for edges to scripts

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                function_calls = find_function_calls(file_path, functions)
                for function_name, calling_file in function_calls.items():
                    G.add_node(function_name, color='lightsteelblue')  # Use a light shade of blue for functions
                    G.add_edge(calling_file, function_name, color='steelblue')  # Use a slightly darker shade of blue for edges to functions
                    G.add_edge(function_name, calling_file, color='steelblue')  # Use a slightly darker shade of blue for reverse edges to functions
                
                # Include class objects as nodes
                for class_name in classes:
                    G.add_node(class_name, color='lightgreen')  # Use a light shade of green for classes
                    G.add_edge(file, class_name, color='green')  # Use green for edges to classes

    return G

if __name__ == '__main__':
    directory_to_scan = 'amworkflow/amworkflow'  # Replace with the directory you want to scan
    dependency_graph = build_dependency_graph(directory_to_scan)

    # Create a pyvis network
    network = Network(height='800px', width='100%', notebook=True, select_menu=True, filter_menu=True)

    # Add nodes and edges to the pyvis network
    for node, data in dependency_graph.nodes(data=True):
        network.add_node(node, color=data.get('color', 'gray'))
    for source, target, data in dependency_graph.edges(data=True):
        network.add_edge(source, target, color=data.get('color', 'gray'))

    # Create an interactive visualization
    network.show_buttons(filter_=['physics'])
    network.toggle_hide_edges_on_drag = True
    network.show('amworkflow/some_thoughts_20230822_new/for_report/dependency_graph.html')
