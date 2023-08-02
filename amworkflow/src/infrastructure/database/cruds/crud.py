from amworkflow.src.infrastructure.database.models.model import db_list
from sqlalchemy import insert
from sqlalchemy.sql.expression import select
import pandas as pd

def insert_data(table: str,
                data: dict, 
                isbatch: bool) -> None:
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.new
    table = db_list[table]
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

def query_data_object(table: str,
               by_name: str,
               column_name: str):
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.new
    if type(table) is str:
        table = db_list[table]
    column = getattr(table, column_name)
    exec_result = session.execute(select(table).filter(column == by_name)).scalar_one()
    return exec_result

def query_multi_data(table: str,
                     by_name: str = None,
                     column_name: str = None,
                     target_column_name: str = None,):
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.new
    table = db_list[table]
    if by_name != None:
        column = getattr(table, column_name)
        if target_column_name != None:
            result = [i.__dict__[target_column_name] for i in session.query(table).filter(column == by_name).all()]
        else:
            result = [i.__dict__ for i in session.query(table).filter(column == by_name).all()]
            for dd in result:
                dd.pop("_sa_instance_state", None)
        
    else:
        exec_result = session.execute(select(table)).all()
        result = [i[0].__dict__ for i in exec_result]
        for dd in result:
            dd.pop("_sa_instance_state", None)
    if target_column_name == None:
        result = pd.DataFrame(result)
    elif len(result) != 0:
        result = result[0]
    return result

def update_data(table: str,
                by_name: str | list,
                target_column: str,
                new_value: int | str | float | bool,
                isbatch: bool) -> None:
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.new
    table = db_list[table]
    if not isbatch:
        transaction = query_data_object(table, by_name, column_name= target_column )
        setattr(transaction, target_column, new_value)
    else:
        for name in by_name:
            transaction = query_data_object(table, name, target_column)
            setattr(transaction, target_column, new_value)
    session.commit()

def delete_data(table: str,
                by_primary_key: str | list,
                isbatch: bool,
                ) -> None:
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.new
    table = db_list[table]
    if not isbatch:
        transaction = session.get(table, by_primary_key)
        session.delete(transaction)
    else:
        for hash in by_primary_key:
            transaction = session.get(table, hash)
            session.delete(transaction)
    session.commit()

def transcation_rollback():
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.rollback()