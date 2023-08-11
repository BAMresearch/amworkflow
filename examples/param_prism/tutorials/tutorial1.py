from amworkflow.src.utils.reader import read_stl_file
from amworkflow.src.geometries.property import get_occ_bounding_box, get_faces, get_face_center_of_mass
from OCC.Core.BRepClass import BRepClass_FaceExplorer
from amworkflow.src.geometries.builder import geometry_builder,sewer
from amworkflow.src.utils.writer import stl_writer
import os


# Read scanned STL file.
stl_path = os.getcwd() + "/map.stl"
print(stl_path)
scan = read_stl_file(filename=stl_path)
#The "scan" here is not yet a "TopoDS_Face" object in PyOCC, but a "TopoDS_Solid" object, which of course consist of faces.
faces = get_faces(scan)
# Using get_faces we can get a list of all faces from the solid object.
xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(scan)
# get the bounding box of this solid object.
# We will get rid of all the model basements in order to only work with the terrain surface. 
bnd_x = [xmin,xmax]
bnd_y = [ymin,ymax]
bnd_z = [zmin]
bnd_face = []
def check(num, list):
    '''
    A small function for checking if the center point of each small face is either on the four basement walls.
    '''
    for elem in list:
       if abs(num - elem) < 1e-3:
           return True
    return False
for face in faces:
    cnt = list(get_face_center_of_mass(face))
    #get center point of each face.
    if check(cnt[0],bnd_x) or check(cnt[1], bnd_y) or check(cnt[2], bnd_z):
        bnd_face.append(face)

filter_face = [face for face in faces if face not in bnd_face]
#filter unwanted faces
sew_face = sewer(filter_face)
#Using sewer to sew all wanted faces into one face, the output is TopoDS_Face
stl_writer(sew_face, "terrain.stl")
#The result will be stored in the amworkflow/src/infrastructure/database/files/output_files. I have pasted the result in our working dir but you are welcome to check for sure.