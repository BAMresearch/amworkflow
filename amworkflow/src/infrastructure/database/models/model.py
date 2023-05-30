from sqlalchemy import create_engine
from src.constants.enums import Directory as D
engine = create_engine("sqlite+pysqlite:////" + D.DATABASE_FILE_PATH.value + r'test.db', echo=True)

from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from src.constants.enums import Timestamp as T
from datetime import datetime
class Base(DeclarativeBase):
    pass

class STLFile(Base):
    __tablename__ = "STLFile"
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
    SliceFile = relationship("SliceFile", cascade="all, delete", back_populates="STLFile")
    MeshFile = relationship("MeshFile", cascade="all, delete", back_populates="STLFile")
    GCode = relationship("GCode", cascade="all, delete", back_populates="STLFile")
    FEResult = relationship("FEResult", cascade="all, delete", back_populates="STLFile")
    

class SliceFile(Base):
    __tablename__ = "SliceFile"
#    slice_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    slice_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    step_length: Mapped[float] = mapped_column(nullable=False)
    stl_hashname_ = mapped_column(ForeignKey('STLFile.stl_hashname', ondelete="CASCADE"))
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    GCode = relationship("GCode", cascade="all, delete", back_populates="SliceFile")
    # gcode_hashname_ = mapped_column(ForeignKey('GCode.gcode_hashname', ondelete="CASCADE"))
    STLFile = relationship("STLFile", back_populates="SliceFile")
    
class MeshFile(Base):
    __tablename__ = "MeshFile"
#    mesh_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    mesh_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    deflection_threshold: Mapped[float] = mapped_column(nullable=False)
    angle_threshold: Mapped[float] = mapped_column(nullable=False)
    tolerance: Mapped[float] = mapped_column(nullable=False)
    iteration: Mapped[int] = mapped_column(nullable=False)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    stl_hashname_ = mapped_column(ForeignKey('STLFile.stl_hashname', ondelete="CASCADE"))
    STLFile = relationship("STLFile", back_populates="MeshFile")
class GCode(Base):
    __tablename__ = "GCode"
#    gcode_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    gcode_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    stl_hashname_ = mapped_column(ForeignKey('STLFile.stl_hashname', ondelete="CASCADE"))
    slice_hashname_ = mapped_column(ForeignKey('SliceFile.slice_hashname', ondelete="CASCADE"))
    STLFile = relationship("STLFile", back_populates="GCode")
    SliceFile = relationship("SliceFile", back_populates="GCode")
    
class FEResult(Base):
    __tablename__ = "FEResult"
#    feresult_id : Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fe_hashname: Mapped[str] = mapped_column(String(32), nullable=False, primary_key=True)
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    batch_num: Mapped[str] = mapped_column(nullable=True)
    stl_hashname_ = mapped_column(ForeignKey('STLFile.stl_hashname', ondelete="CASCADE"))
    STLFile = relationship("STLFile", back_populates="FEResult")
    
