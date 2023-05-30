from src.infrastructure.database.models.model import engine, Base
from sqlalchemy.orm import Session

Base.metadata.create_all(engine)
session = Session(engine)