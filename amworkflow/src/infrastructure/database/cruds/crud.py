from uuid import uuid4
from amworkflow.src.infrastructure.database.Engine.engine import session
from sqlalchemy import insert
from sqlalchemy.sql.expression import select

def insert_data(table: callable,
                data: dict, 
                isbatch: bool) -> None:
    session.new
    try:
        if not isbatch:
            transaction = table(**data)
            session.add(transaction)
        else:
            for sub_data in data:
                transaction = table(**sub_data)
                session.add(transaction)
    except:
        transcation_rollback()  
    session.commit()

def _query_data(table: callable,
               by_hash: str):
    session.new
    exec_result = session.execute(select(table).filter_by(stl_hashname = by_hash)).scalar_one()
    return exec_result

def query_multi_data(table: callable,
                     by_name: str = None,
                     column_name: str = None,
                     target_column_name: str = None):
    session.new
    if by_name != None:
        column = getattr(table, column_name)
        result = [i.__dict__ for i in session.query(table).filter(column == by_name).all()]
        if target_column_name != None:
            result = [i.__dict__[target_column_name] for i in session.query(table).filter(column == by_name).all()]
        return result
    else:
        exec_result = session.execute(select(table))
        return exec_result.all()

def update_data(table: callable,
                by_hash: str | list,
                target_column: str,
                new_value: int | str | float | bool,
                isbatch: bool) -> None:
    session.new
    if not isbatch:
        transaction = _query_data(table, by_hash)
        setattr(transaction, target_column, new_value)
    else:
        for hash in by_hash:
            transaction = _query_data(table, hash)
            setattr(transaction, target_column, new_value)
    session.commit()

def delete_data(table: callable,
                by_hash: str | list,
                isbatch: bool,
                column: str | list = None) -> None:
    session.new
    if not isbatch:
        transaction = session.get(table, by_hash)
        session.delete(transaction)
    else:
        for hash in by_hash:
            transaction = session.get(table, hash)
            session.delete(transaction)
    session.commit()

def transcation_rollback():
    session.rollback()