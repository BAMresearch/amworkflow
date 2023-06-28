from sqlalchemy import create_engine
from src.constants.enums import Directory as D
engine = create_engine("sqlite+pysqlite:////" + D.DATABASE_FILE_PATH.value + r'amworkflow.db', echo=True)

from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from src.constants.enums import Timestamp as T
from datetime import datetime
class Base(DeclarativeBase):
    pass

class GeometryFile(Base):
    __tablename__ = "GeometryFile"
#    stl_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String)
    stl_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    withCurve: Mapped[bool] = mapped_column(nullable=False)
    length: Mapped[float] = mapped_column(nullable=False)
    width: Mapped[float] = mapped_column(nullable=False)
    height: Mapped[float] = mapped_column(nullable=False)
    radius: Mapped[float]
    linear_deflection: Mapped[float] = mapped_column(nullable=False)
    angular_deflection: Mapped[float] = mapped_column(nullable=False)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    SliceFile = relationship("SliceFile", cascade="all, delete", back_populates="GeometryFile")
    XdmfFile = relationship("XdmfFile", cascade="all, delete", back_populates="GeometryFile")
    GCode = relationship("GCode", cascade="all, delete", back_populates="GeometryFile")
    FEResult = relationship("FEResult", cascade="all, delete", back_populates="GeometryFile")
    

class SliceFile(Base):
    __tablename__ = "SliceFile"
#    slice_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    slice_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    step_length: Mapped[float] = mapped_column(nullable=False)
    stl_hashname = mapped_column(ForeignKey('GeometryFile.stl_hashname', ondelete="CASCADE"))
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
    stl_hashname = mapped_column(ForeignKey('GeometryFile.stl_hashname', ondelete="CASCADE"))
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
    stl_hashname = mapped_column(ForeignKey('GeometryFile.stl_hashname', ondelete="CASCADE"))
    slice_hashname_ = mapped_column(ForeignKey('SliceFile.slice_hashname', ondelete="CASCADE"))
    GeometryFile = relationship("GeometryFile", back_populates="GCode")
    SliceFile = relationship("SliceFile", back_populates="GCode")
    
class FEResult(Base):
    __tablename__ = "FEResult"
#    feresult_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fe_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    stl_hashname = mapped_column(ForeignKey('GeometryFile.stl_hashname', ondelete="CASCADE"))
    GeometryFile = relationship("GeometryFile", back_populates="FEResult")
    
