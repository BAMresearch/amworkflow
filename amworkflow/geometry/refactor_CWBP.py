import multiprocessing

multiprocessing.set_start_method("fork")
import matplotlib.pyplot as plt
import numpy as np

import amworkflow.geometry.builtinCAD as bcad


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
        self.digraph_points = {}
        self.init_points(self.coords, {"position": "center", "active": True})
        self.index_points = []
        self.modify_edges = {}
        self.create_sides()

    def init_points(self, points: list, prop: dict = None):
        for i, p in enumerate(points):
            if not isinstance(p, bcad.Pnt):
                p = bcad.Pnt(p)
            self.enrich_component(p, prop.copy())
            self.center_pnts.append(p)

    def update_index_points(self):
        index_points = []
        for i in bcad.id_index.values():
            if i["type"] == 0 and "CWBP" in i["point"].property:
                if i["point"].property["CWBP"]["active"]:
                    index_points.append(i["id"])
        self.index_points = index_points

    def enrich_component(
        self, component: bcad.TopoObj, prop: dict = {}
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
                self.center_pnts[index_n], self.center_pnts[index]
            )
            self.enrich_component(
                seg_current_next, {"position": "center", "active": True}
            )
            # The segment from the previous index to the current index
            seg_previous_current = bcad.Segment(
                self.center_pnts[index], self.center_pnts[index_l]
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
            seg_side = bcad.Segment(self.side_pnts[i + 1], self.side_pnts[i])
            self.enrich_component(seg_side, {"position": "side", "active": True})
            self.side_segments.append(seg_side)

    def update_digraph(
        self,
        start_node: int,
        end_node: int,
        insert_node: int = None,
        build_new_edge: bool = True,
    ) -> None:
        self.update_index_points()
        if start_node not in self.index_points:
            raise Exception(f"Unrecognized start node: {start_node}.")
        if end_node not in self.index_points:
            raise Exception(f"Unrecognized end node: {end_node}.")
        if (insert_node not in self.index_points) and (insert_node is not None):
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
        for line1 in cwbp.side_segments:
            for line2 in cwbp.side_segments:
                if line1.id != line2.id:
                    if [line1.id, line2.id] not in visited or [
                        line2.id,
                        line1.id,
                    ] not in visited:
                        visited.append([line1.id, line2.id])
                        line_pairs.append((line1, line2))
        num_processes = multiprocessing.cpu_count()
        # chunk_size = len(line_pairs) // num_processes
        # chunks = [
        #     line_pairs[i : i + chunk_size]
        #     for i in range(0, len(line_pairs), chunk_size)
        # ]
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.starmap(bcad.find_intersect_node_on_edge, line_pairs)
            for result in results:
                if result is not None:
                    for pair in result:
                        print(pair)
                        self.modify_edges.update({pair[0]: pair[1]})


n = 20
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
# cwbp.update_index_points()
# print(cwbp.index_points)
# bcad.id_index[0]["point"].property["CWBP"]["active"] = False
# cwbp.update_index_points()
# print(cwbp.index_points)
# print(bcad.id_index)
# line1, line2 = cwbp.side_segments[0], cwbp.side_segments[1]
# result = bcad.find_intersect_node_on_edge(line1, line2)
# print(result)
cwbp.find_overlap_node_on_edge_parallel()
from pprint import pprint

pprint(cwbp.modify_edges)


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
