from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from amworkflow.src.constants.enums import Timestamp as T
from datetime import datetime
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
    batch_num: Mapped[str] = mapped_column(nullable=True)
    SliceFile = relationship("SliceFile", cascade="all, delete", back_populates="GeometryFile")
    XdmfFile = relationship("XdmfFile", cascade="all, delete", back_populates="GeometryFile")
    GCode = relationship("GCode", cascade="all, delete", back_populates="GeometryFile")
    FEResult = relationship("FEResult", cascade="all, delete", back_populates="GeometryFile")
    ModelProfile = relationship("ModelProfile", back_populates="GeometryFile")
    

class SliceFile(Base):
    __tablename__ = "SliceFile"
#    slice_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    slice_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    step_length: Mapped[float] = mapped_column(nullable=False)
    geom_hashname = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    GCode = relationship("GCode", cascade="all, delete", back_populates="SliceFile")
    # gcode_hashname_ = mapped_column(ForeignKey('GCode.gcode_hashname', ondelete="CASCADE"))
    GeometryFile = relationship("GeometryFile", back_populates="SliceFile")
    
class XdmfFile(Base):
    __tablename__ = "XdmfFile"
#    mesh_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(nullable=False)
    xdmf_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    mesh_size_factor: Mapped[float] = mapped_column(nullable=False)
    layer_thickness: Mapped[float] = mapped_column(nullable=True)
    layer_num: Mapped[int] = mapped_column(nullable=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    geom_hashname = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    GeometryFile = relationship("GeometryFile", back_populates="XdmfFile")
    H5File = relationship("H5File", cascade="all, delete", back_populates="XdmfFile")
    
class H5File(Base):
    __tablename__ = "H5File"
#    mesh_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(nullable=False)
    h5_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    xdmf_hashname = mapped_column(ForeignKey('XdmfFile.xdmf_hashname', ondelete="CASCADE"))
    XdmfFile = relationship("XdmfFile", back_populates="H5File")
    
class GCode(Base):
    __tablename__ = "GCode"
#    gcode_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    gcode_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
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
    ModelParameter = relationship("ModelParameter", cascade="all, delete", back_populates="ModelProfile")
    GeometryFile = relationship("GeometryFile", cascade="all, delete", back_populates="ModelProfile")

class ModelParameter(Base):
    __tablename__ = "ModelParameter"
    param_name: Mapped[str] = mapped_column(nullable=False, primary_key=True)
    model_name: Mapped[str] = mapped_column(ForeignKey('ModelProfile.model_name', ondelete="CASCADE"))
    ModelProfile = relationship("ModelProfile", back_populates="ModelParameter")
    param_type: Mapped[str] = mapped_column(ForeignKey('ParameterType.type_name', ondelete="CASCADE"))

class ParameterValue(Base):
    __tablename__ = "ParameterValue"
    value_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    param_name: Mapped[str] = mapped_column(ForeignKey('ModelParameter.param_name', ondelete="CASCADE"))
    geom_hashname: Mapped[str] = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    param_value: Mapped[float] = mapped_column(nullable=True)
    
class ParameterType(Base):
    __tablename__ = "ParameterType"
    type_name: Mapped[str] = mapped_column(nullable=False, primary_key=True)

class IterationParameter(Base):
    __tablename__ = "IterationParameter"
    endpoint: Mapped[float] = mapped_column(nullable=True)
    num: Mapped[int] = mapped_column(nullable=True, default=0)
    geom_hashname: Mapped[str] = mapped_column(ForeignKey('GeometryFile.geom_hashname', ondelete="CASCADE"))
    parameter_name: Mapped[str] = mapped_column(ForeignKey('ModelParameter.param_name', ondelete="CASCADE"))
    iter_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    
    
    

