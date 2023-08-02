import streamlit as st
import amworkflow.src.infrastructure.database.engine.config as cfg
from amworkflow.src.infrastructure.database.cruds.crud import query_multi_data
cfg.DB_DIR = "/home/yhe/Documents/amworkflow/usecases/param_wall/db"
dt = query_multi_data("ModelProfile")
st.write("""
# My first app
Hello *world!*
""")
st.write(dt)
