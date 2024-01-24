from amworkflow.src.infrastructure.database.models.model import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from amworkflow.src.infrastructure.database.engine.config import DB_DIR
engine = create_engine("sqlite+pysqlite:////" + DB_DIR + r'/amworkflow.db') #echo = True for getting logging
Base.metadata.create_all(engine)
session = Session(engine)
