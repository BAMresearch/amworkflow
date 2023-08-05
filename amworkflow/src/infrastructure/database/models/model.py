import sys
import inspect
from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String, Column, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from amworkflow.src.constants.enums import Timestamp as T
from datetime import datetime

current_module = sys.modules[__name__]
db_list = {}

class Base(DeclarativeBase):
    pass

class GeometryFile(Base):
    __tablename__ = "GeometryFile"
#    stl_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String)
    geom_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    model_name: Mapped[str] = mapped_column(ForeignKey('ModelProfile.model_name', ondelete="CASCADE"))
    linear_deflection: Mapped[float] = mapped_column(nullable=True)
    angular_deflection: Mapped[float] = mapped_column(nullable=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    is_imported: Mapped[bool] = mapped_column(default=False)
    task_id: Mapped[str] = mapped_column(ForeignKey("Task.task_id", ondelete="CASCADE"))
    SliceFile = relationship("SliceFile", cascade="all, delete", back_populates="GeometryFile")
    MeshFile = relationship("MeshFile", cascade="all, delete", back_populates="GeometryFile")
    GCode = relationship("GCode", cascade="all, delete", back_populates="GeometryFile")
    FEResult = relationship("FEResult", cascade="all, delete", back_populates="GeometryFile")
    ModelProfile = relationship("ModelProfile", back_populates="GeometryFile")
    Task = relationship("Task", back_populates = "GeometryFile")
    

class SliceFile(Base):
    __tablename__ = "SliceFile"
#    slice_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    slice_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    step_length: Mapped[float] = mapped_column(nullable=False)
    geom_hashname = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    task_id: Mapped[str] = mapped_column(nullable=True)
    GCode = relationship("GCode", cascade="all, delete", back_populates="SliceFile")
    # gcode_hashname_ = mapped_column(ForeignKey('GCode.gcode_hashname', ondelete="CASCADE"))
    GeometryFile = relationship("GeometryFile", back_populates="SliceFile")
    
class MeshFile(Base):
    __tablename__ = "MeshFile"
#    mesh_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(nullable=False)
    mesh_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    mesh_size_factor: Mapped[float] = mapped_column(nullable=False)
    layer_thickness: Mapped[float] = mapped_column(nullable=True)
    layer_num: Mapped[int] = mapped_column(nullable=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    task_id: Mapped[str] = mapped_column(ForeignKey("Task.task_id", ondelete="CASCADE"))
    geom_hashname = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    GeometryFile = relationship("GeometryFile", back_populates="MeshFile")
    Task = relationship("Task", back_populates = "MeshFile")
    
class GCode(Base):
    __tablename__ = "GCode"
#    gcode_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    gcode_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    task_id: Mapped[str] = mapped_column(nullable=True)
    geom_hashname = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    slice_hashname = mapped_column(ForeignKey('SliceFile.slice_hashname', ondelete="CASCADE"))
    GeometryFile = relationship("GeometryFile", back_populates="GCode")
    SliceFile = relationship("SliceFile", back_populates="GCode")
    
class FEResult(Base):
    __tablename__ = "FEResult"
#    feresult_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fe_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    geom_hashname = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    GeometryFile = relationship("GeometryFile", back_populates="FEResult")

class ModelProfile(Base):
    __tablename__ = "ModelProfile"
    model_name: Mapped[str] = mapped_column(nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    GeometryFile = relationship("GeometryFile", cascade="all, delete", back_populates="ModelProfile")
    imported_file_id: Mapped[str] = mapped_column(ForeignKey('ImportedFile.md5_id', ondelete="CASCADE"), nullable=True)
    ImportedFile = relationship("ImportedFile", back_populates="ModelProfile")
    ParameterToProfile = relationship("ParameterToProfile", cascade="all, delete", back_populates="ModelProfile")
    Task = relationship("Task", back_populates="ModelProfile", cascade="all, delete")

class ModelParameter(Base):
    __tablename__ = "ModelParameter"
    param_name: Mapped[str] = mapped_column(nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    ParameterToProfile = relationship("ParameterToProfile", cascade="all, delete", back_populates="ModelParameter")

class ParameterToProfile(Base):
    __tablename__ = "ParameterToProfile"
    # id = Column(Integer, primary_key=True, autoincrement=True)
    param_name: Mapped[str] = mapped_column(ForeignKey('ModelParameter.param_name', ondelete="CASCADE"),nullable=False,primary_key=True)
    model_name: Mapped[str] = mapped_column(ForeignKey('ModelProfile.model_name', ondelete="CASCADE"), primary_key=True)
    param_status: Mapped[bool] = mapped_column(nullable=False, default=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    # model_parameter = relationship("ModelParameter", foreign_keys=[param_name])
    # profile_name = relationship("ModelProfile", foreign_keys=[model_name])
    ModelParameter = relationship("ModelParameter", back_populates="ParameterToProfile")
    ModelProfile = relationship("ModelProfile", back_populates="ParameterToProfile")

class ParameterValue(Base):
    __tablename__ = "ParameterValue"
    value_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    param_name: Mapped[str] = mapped_column(ForeignKey('ModelParameter.param_name', ondelete="CASCADE"))
    geom_hashname: Mapped[str] = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    param_value: Mapped[float] = mapped_column(nullable=True)
    
# class ParameterType(Base):
#     __tablename__ = "ParameterType"
#     type_name: Mapped[str] = mapped_column(nullable=False, primary_key=True)

# class IterationParameter(Base):
#     __tablename__ = "IterationParameter"
#     endpoint: Mapped[float] = mapped_column(nullable=True)
#     num: Mapped[int] = mapped_column(nullable=True, default=0)
#     geom_hashname: Mapped[str] = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
#     parameter_name: Mapped[str] = mapped_column(ForeignKey('ModelParameter.param_name', ondelete="CASCADE"))
#     iter_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    
class ImportedFile(Base):
    __tablename__ = "ImportedFile"
    filename:  Mapped[str] = mapped_column(nullable=False)
    md5_id: Mapped[str] = mapped_column(nullable=False, primary_key=True)
    ModelProfile = relationship("ModelProfile", cascade="all, delete", back_populates="ImportedFile")
    
class Task(Base):
    __tablename__ = "Task"
    task_id: Mapped[str] = mapped_column(nullable=False, primary_key=True)
    model_name: Mapped[str] = mapped_column(ForeignKey('ModelProfile.model_name', ondelete="CASCADE"))
    stl: Mapped[bool] = mapped_column(default=True)
    stp: Mapped[bool] = mapped_column(default=False)
    xdmf: Mapped[bool] = mapped_column(default=False)
    h5: Mapped[bool] = mapped_column(default=False)
    vtk: Mapped[bool] = mapped_column(default=False)
    msh: Mapped[bool] = mapped_column(default=False)
    ModelProfile = relationship("ModelProfile", back_populates="Task")
    MeshFile = relationship("MeshFile", back_populates="Task", cascade="all, delete")
    GeometryFile = relationship("GeometryFile", back_populates="Task", cascade="all, delete")
    

for name, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj):
            db_list.update({name: obj})

