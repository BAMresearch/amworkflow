import streamlit as st
import pyvista as pv
from stpyvista import stpyvista

temp_dir = "/home/yuxiang/Documents/BAM/amworkflow/"
file_dir = temp_dir + "/"+"123.stl"
reader = pv.STLReader(file_dir)
mesh = reader.read()
plotter = pv.Plotter(border=False, window_size=[500, 400])
plotter.background_color = "#f0f8ff"
plotter.add_mesh(mesh, color="white")
plotter.view_isometric()
placeholder = st.empty()
with placeholder.container():
    stpyvista(plotter, key="my_stl")
