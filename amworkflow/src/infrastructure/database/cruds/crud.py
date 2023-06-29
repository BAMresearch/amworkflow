from uuid import uuid4
from amworkflow.src.infrastructure.database.Engine.engine import session
from sqlalchemy import insert
from sqlalchemy.sql.expression import select

def insert_data(table: callable,
                data: dict, 
                isbatch: bool) -> None:
    session.new
    if not isbatch:
        transaction = table(**data)
        session.add(transaction)
    else:
        for sub_data in data:
            transaction = table(**sub_data)
            session.add(transaction)  
    session.commit()

def _query_data(table: callable,
               by_hash: str):
    exec_result = session.execute(select(table).filter_by(stl_hashname = by_hash)).scalar_one()
    return exec_result

def query_multi_data(table: callable,
                     by_batch_num: str):
    if by_batch_num != None:
        exec_result = session.execute(select(table).filter_by(batch_num = by_batch_num))
    else:
        exec_result = session.execute(select(table))
    return exec_result

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
