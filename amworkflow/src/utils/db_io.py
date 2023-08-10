from amworkflow.src.infrastructure.database.cruds.crud import query_multi_data
from amworkflow.src.constants.enums import Timestamp as T
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
        query = query_multi_data("GeometryFile", t_id, "task_id")
        query_t = query_multi_data("Task", by_name=t_id, column_name = "task_id")
        if query_t.empty:
            print(f"\033[1mTask {t_id} not found.\033[0m")
        else:
            path = os.path.join(file_dir, t_id)
            dest = mk_dir(output_dir, t_id)
            stp = query_t.stp[0]
            msh = query_t.msh[0]
            vtk = query_t.vtk[0]
            if query.empty:
                print(f"\033[1mFile in task {t_id} not found.\033[0m")
            else:
                hash = query.geom_hashname.to_list()
                name = query.filename.to_list()
                if org:
                    geom_p = mk_dir(dest, "geometry_file")
                    dest_g = geom_p
                for i, hh in enumerate(hash):
                    file_p = os.path.join(path, hh+".stl")
                    dest_p = os.path.join(dest_g,name[i]+".stl")
                    if stp:
                        file_pp = os.path.join(path, hh+".stp")
                        dest_pp = os.path.join(dest_g, name[i]+".stp")
                        copyer(file_pp, dest_pp)
                    copyer(file_p, dest_p)
            query_m = query_multi_data("MeshFile", t_id, "task_id")
            if query_m.empty:
                print(f"\033[1mno mesh file in task {t_id} found, skip...\033[0m")
            else:
                if org:
                    mesh_dir = mk_dir(dest, "mesh_file")
                    dest_m = mesh_dir
                for ind, row in query_m.iterrows():
                    if vtk:
                        file_pp = os.path.join(path, row.mesh_hashname+".vtk")
                        dest_pp = os.path.join(dest_m, row.filename+".vtk")
                        copyer(file_pp, dest_pp)
                    if msh:
                        file_pp = os.path.join(path, row.mesh_hashname+".msh")
                        dest_pp = os.path.join(dest_m, row.filename+".msh")
                        copyer(file_pp, dest_pp)
                    file_p = os.path.join(path, row.mesh_hashname+".xdmf")
                    dest_p = os.path.join(dest_m,row.filename+".xdmf")
                    file_p2 = os.path.join(path, row.mesh_hashname+".h5")
                    dest_p2 = os.path.join(dest_m,row.filename+".h5")
                    copyer(file_p, dest_p)
                    copyer(file_p2, dest_p2)
            print(f"\033[1mTask {t_id} Downloaded successfully.\033[0m")
    if time_range is not None:
        st = convert_to_datetime(time_range[0])
        nd = convert_to_datetime(time_range[1])
        query_t = query_multi_data("Task").task_id.to_list()
        t_id_l = [convert_to_datetime(i) for i in query_t]
        filtered = [i for i in t_id_l if i <= nd and i > st]
        filtered_s = [i.strftime(T.YY_MM_DD_HH_MM_SS.value) for i in filtered]
        print(filtered_s)
        for tk in filtered_s:
            print(tk)
            worker(tk)
    if task_id is not None:
        if task_id == "all":
            query_t = query_multi_data("Task").task_id.to_list()
            for tk in query_t:
                worker(tk)
        else:
            tid = convert_to_datetime(task_id).strftime(T.YY_MM_DD_HH_MM_SS.value)
            worker(tid)

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
    
def delete_dir(dir_p: str):
    file_l = os.listdir(dir_p)
    if len(file_l) == 0:
        os.rmdir(dir_p)
    else:
        shutil.rmtree(dir_p)

def file_delete(dir_path: str, filename: str = None, operate: str = None, op_list: list = None):
    path_valid_check(dir_path)
    if filename is not None:
        file_p = os.path.join(dir_path, filename)
        if os.path.isfile(file_p):
            os.remove(file_p)
        else:
            print(f"{file_p} not invalid.")
    if operate is not None:
        if operate == "all":
           delete_dir(dir_path)
        elif operate == "list":
            for item in op_list:
                p = os.path.join(dir_path, item)
                if os.path.isfile(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    delete_dir(p)
