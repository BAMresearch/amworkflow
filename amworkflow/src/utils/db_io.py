from amworkflow.src.infrastructure.database.cruds.crud import query_multi_data
from amworkflow.src.constants.enums import Directory as D
from amworkflow.src.utils.sanity_check import path_valid_check
from amworkflow.src.utils.writer import mk_dir
import shutil
import os
from amworkflow.src.utils.writer import convert_to_datetime
#TODO: rewrite downloader

def downloader(file_dir: str,
               output_dir: str,
               task_id: int | str = None,
               time_range: list = None,
               org: bool = True):
    def worker(t_id: str):
        path = os.path.join(file_dir, t_id)
        dest = mk_dir(output_dir, t_id)
        query = query_multi_data("GeometryFile", t_id, "task_id")
        if query.empty:
            print(f"Task {t_id} not found.")
        else:
            hash = query.geom_hashname.to_list
            name = query.filename.to_list
            stp = query.stp.to_list
            if org:
                geom_p = mk_dir(dest, "geometry_file")
                dest = geom_p
            for i, hh in enumerate(hash):
                file_p = os.path.join(path, hh+".stl")
                dest_p = os.path.join(dest,name[i]+".stl")
                if stp[i] == 1:
                    file_pp = os.path.join(path, hh+".stp")
                    dest_pp = os.path.join(dest, name[i]+".stp")
                    copyer(file_pp, dest_pp)
                copyer(file_p, dest_p)
        query_m = query_multi_data("MeshFile", task_id, "task_id")
        if query_m.empty:
            print(f"no mesh file in task {task_id} found, skip...")
        else:
            if org:
                mesh_dir = mk_dir(dest, "mesh_file")
                dest = mesh_dir
            for ind, row in query_m.iterrows():
                if row.vtk == 1:
                    file_pp = os.path.join(path, row.mesh_hashname+".vtk")
                    dest_pp = os.path.join(dest, row.filename+".vtk")
                    copyer(file_pp, dest_pp)
                if row.msh == 1:
                    file_pp = os.path.join(path, row.mesh_hashname+".msh")
                    dest_pp = os.path.join(dest, row.filename+".msh")
                    copyer(file_pp, dest_pp)
                file_p = os.path.join(path, row.mesh_hashname+".xdmf")
                dest_p = os.path.join(dest,row.mesh_hashname+".xdmf")
                file_p2 = os.path.join(path, row.mesh_hashname+".h5")
                dest_p2 = os.path.join(dest,row.mesh_hashname+".h5")
                copyer(file_p, dest_p)
                copyer(file_p2, dest_p2)
    st = convert_to_datetime(time_range[0])
    nd = convert_to_datetime(time_range[1])
    query_multi_data("Geomet")
        #TODO: select time

def copyer(src_file: str, dest_file: str):
    shutil.copy2(src_file, dest_file)
    
def mover(src: str, dest: str):
    shutil.move(src=src, dst=dest)
    
def file_copy(path1: str, path2: str) -> bool:
    path_valid_check(path1)
    path_valid_check(path2)
    try:
        shutil.copy(path1, path2)
        return True
    except:
        return False