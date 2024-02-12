import matplotlib.pyplot as plt
import numpy as np

import amworkflow.geometry.builtinCAD as bcad

# multiprocessing.set_start_method("fork")


class CreateWallByPoints:
    def __init__(self, coords: list, th: float, height: float, is_close: bool = True):
        self.coords = coords
        self.height = height
        self.th = th
        self.is_close = is_close
        self.volume = 0
        self.center_pnts = []
        self.side_pnts = []
        self.center_segments = []
        self.side_segments = []
        self.left_pnts = []
        self.right_pnts = []
        self.left_segments = []
        self.right_segments = []
        self.imaginary_pnt = {}
        self.imaginary_segments = {}
        self.digraph_points = {}
        self.init_points(self.coords, {"position": "center", "active": True})
        self.index_components = {}
        self.edges_to_be_modified = {}
        self.create_sides()
        self.find_overlap_node_on_edge_parallel()
        self.modify_edge()

    def init_points(self, points: list, prop: dict = None):
        for i, p in enumerate(points):
            if not isinstance(p, bcad.Pnt):
                p = bcad.Pnt(p)
            self.enrich_component(p, prop.copy())
            self.center_pnts.append(p)

    def update_index_component(self):
        for i in bcad.component_lib.values():
            if "CWBP" not in i[bcad.TYPE_INDEX[i["type"]]].property:
                continue
            property_name = bcad.TYPE_INDEX[i["type"]]
            if i[property_name].property["CWBP"]["active"]:
                if i[property_name].id not in self.index_components:
                    self.index_components.update({property_name: [i[property_name].id]})
                else:
                    self.index_component[property_name].append(i[property_name].id)

    def enrich_component(
        self, component: bcad.TopoObj, prop: dict = None
    ) -> bcad.TopoObj:
        if not isinstance(component, bcad.TopoObj):
            raise ValueError("Component must be a TopoObj")
        component.enrich_property({"CWBP": prop})

    def compute_index(self, ind, length):
        return ind % length if ind > 0 else -(-ind % length)

    def compute_support_vector(self, seg1: bcad.Segment, seg2: bcad.Segment = None):
        lit_vector = bcad.get_literal_vector(seg1.vector, True)
        if seg2 is None:
            th = self.th * 0.5
            angle = np.pi / 2
            sup_vector = lit_vector
        else:
            sup_vector = bcad.bisect_angle(-seg2.vector, seg1)
            angle = bcad.angle_of_two_arrays(lit_vector, sup_vector)
            if angle > np.pi / 2:
                sup_vector *= -1
                angle = np.pi - angle
            th = np.abs(self.th * 0.5 / np.cos(angle))
        return sup_vector, th

    def create_sides(self):
        close = 1 if self.is_close else 0
        for i in range(len(self.center_pnts) + close):
            # The previous index
            index_l = self.compute_index(i - 1, len(self.center_pnts))
            # The current index
            index = self.compute_index(i, len(self.center_pnts))
            # The next index
            index_n = self.compute_index(i + 1, len(self.center_pnts))
            # The segment from the current index to the next index
            seg_current_next = bcad.Segment(
                self.center_pnts[index], self.center_pnts[index_n]
            )
            self.enrich_component(
                seg_current_next, {"position": "center", "active": True}
            )
            # The segment from the previous index to the current index
            seg_previous_current = bcad.Segment(
                self.center_pnts[index_l], self.center_pnts[index]
            )
            self.enrich_component(
                seg_previous_current, {"position": "center", "active": True}
            )
            if index == 0 and not self.is_close:
                seg1 = seg_current_next
                seg2 = None
            elif index == len(self.center_pnts) - 1 and not self.is_close:
                seg1 = seg_previous_current
                seg2 = None
            else:
                seg1 = seg_current_next
                seg2 = seg_previous_current
            su_vector, su_len = self.compute_support_vector(seg1, seg2)
            lft_pnt = bcad.Pnt(su_vector * su_len + self.center_pnts[index].coord)
            rgt_pnt = bcad.Pnt(-su_vector * su_len + self.center_pnts[index].coord)
            self.left_pnts.append(lft_pnt)
            self.enrich_component(lft_pnt, {"position": "left", "active": True})
            self.right_pnts.append(rgt_pnt)
            self.enrich_component(rgt_pnt, {"position": "right", "active": True})
        self.right_pnts = self.right_pnts[::-1]
        self.side_pnts = self.left_pnts + self.right_pnts
        for i in range(len(self.side_pnts) - 1):
            seg_side = bcad.Segment(self.side_pnts[i], self.side_pnts[i + 1])
            self.enrich_component(seg_side, {"position": "side", "active": True})
            self.side_segments.append(seg_side)
        self.update_index_component()

    def update_digraph(
        self,
        start_node: int,
        end_node: int,
        insert_node: int = None,
        build_new_edge: bool = True,
    ) -> None:
        self.update_index_component()
        if start_node not in self.index_components:
            raise Exception(f"Unrecognized start node: {start_node}.")
        if end_node not in self.index_components:
            raise Exception(f"Unrecognized end node: {end_node}.")
        if (insert_node not in self.index_components) and (insert_node is not None):
            raise Exception(f"Unrecognized inserting node: {insert_node}.")
        if start_node in self.digraph_points:
            # Directly update the end node
            if insert_node is None:
                self.digraph_points[start_node].append(end_node)
            # Insert a new node between the start and end nodes
            # Basically, the inserted node replaces the end node
            # If build new edge is True, a new edge consists of inserted node and the new end node will be built.
            else:
                end_node_list_index = self.digraph_points[start_node].index(end_node)
                self.digraph_points[start_node][end_node_list_index] = insert_node
                if build_new_edge:
                    edge1 = bcad.Segment(
                        bcad.component_lib[start_node], bcad.component_lib[insert_node]
                    )
                    self.enrich_component(
                        edge1, {"position": "digraph", "active": True}
                    )
                    edge2 = bcad.Segment(
                        bcad.component_lib[insert_node], bcad.component_lib[end_node]
                    )
                    self.enrich_component(
                        edge2, {"position": "digraph", "active": True}
                    )
                    self.digraph_points.update({insert_node: [end_node]})
        else:
            # If there is no edge for the start node, a new edge will be built.
            if insert_node is None:
                self.digraph_points.update({start_node: [end_node]})
            else:
                raise Exception("No edge found for insertion option.")

    def find_overlap_node_on_edge_parallel(self):
        visited = []
        line_pairs = []
        for line1 in self.side_segments:
            for line2 in self.side_segments:
                if line1.id != line2.id:
                    if [line1.id, line2.id] not in visited or [
                        line2.id,
                        line1.id,
                    ] not in visited:
                        visited.append([line1.id, line2.id])
                        visited.append([line2.id, line1.id])
                        line_pairs.append((line1, line2))
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.starmap(bcad.find_intersect_node_on_edge, line_pairs)
            for result in results:
                if result is not None:
                    for pair in result:
                        if pair[0] not in self.edges_to_be_modified:
                            self.edges_to_be_modified.update({pair[0]: [pair[1]]})
                        else:
                            self.edges_to_be_modified[pair[0]].append(pair[1])

    def modify_edge(self):
        for edge_id, nodes_id in self.edges_to_be_modified.items():
            edge = bcad.component_lib[edge_id]["segment"]
            nodes = [bcad.component_lib[i]["point"] for i in nodes_id]
            edge_0_coords = edge.raw_value[0]
            nodes_coords = [node.value for node in nodes]
            distances = [np.linalg.norm(i - edge_0_coords) for i in nodes_coords]
            order = np.argsort(distances)
            nodes = [nodes[i] for i in order]
            bcad.component_lib[edge_id]["segment"].property["CWBP"]["active"] = False
            pts_list = [edge.value[0]] + nodes + [edge.value[1]]
            for i, nd in enumerate(pts_list):
                if i == 0:
                    continue
                self.update_digraph(pts_list[i - 1], nd)
                if (i != 1) and (i != len(pts_list) - 1):
                    if (pts_list[i - 1] in self.imaginary_pnt) and nd in (
                        self.imaginary_pnt
                    ):
                        self.imaginary_segments.update({pts_list[i - 1]: nd})


n = 5
pnt_set = [bcad.get_random_pnt(0, 10, 0, 50, 0, 0, numpy_array=False) for i in range(n)]
# pnt_set = [bcad.Pnt([0, 0]), bcad.Pnt([2, 0]), bcad.Pnt([2, 2])]
cwbp = CreateWallByPoints(pnt_set, 0.5, 0, True)
lft_coords = [i.coord for i in cwbp.left_pnts]
rgt_coords = [i.coord for i in cwbp.right_pnts]
x_lft_coords = [i[0] for i in lft_coords]
y_lft_coords = [i[1] for i in lft_coords]
x_rgt_coords = [i[0] for i in rgt_coords]
y_rgt_coords = [i[1] for i in rgt_coords]
x_coords = x_lft_coords + x_rgt_coords
y_coords = y_lft_coords + y_rgt_coords


# print(cwbp.left_pnts[1].property["CWBP"])
# cwbp.update_index_component()
# print(cwbp.index_points)
# bcad.component_lib[0]["point"].property["CWBP"]["active"] = False
# cwbp.update_index_component()
# print(cwbp.index_points)
# print(bcad.component_lib)
# line1, line2 = cwbp.side_segments[0], cwbp.side_segments[1]
# result = bcad.find_intersect_node_on_edge(line1, line2)
# print(result)
# cwbp.find_overlap_node_on_edge_parallel()
from pprint import pprint

# pprint(cwbp.edges_to_be_modified)
# pprint(cwbp.digraph_points)
pprint(cwbp.index_points)
pprint(cwbp.index_segments)


def plot_pnts(x, y):
    # Create a 2D plot
    plt.figure()

    # Plot the points
    plt.scatter(x, y)

    plt.plot(x, y, linestyle="-", marker="", color="blue")
    # Set labels
    plt.xlabel("X Label")
    plt.ylabel("Y Label")

    # Optionally, you can add a title
    plt.title("Random 2D Points")

    # Show the plot
    plt.show()


# plot_pnts(x_coords, y_coords)
# if __name__ == "__main__":
#     main()
#     from pprint import pprint

#     pprint(stored)
