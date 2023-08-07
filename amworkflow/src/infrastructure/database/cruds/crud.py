from amworkflow.src.infrastructure.database.models.model import db_list
from sqlalchemy import insert
from sqlalchemy.sql.expression import select
import pandas as pd
from amworkflow.src.utils.visualizer import color_background

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
    exec_result = session.execute(select(table).filter(column == by_name)).all()
    return exec_result

def query_multi_data(table: str,
                     by_name: str = None,
                     column_name: str = None,
                     snd_by_name: str = None,
                     snd_column_name: str = None,
                     target_column_name: str = None,):
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.new
    table = db_list[table]
    if by_name != None:
        column = getattr(table, column_name)
        if snd_column_name is not None:
            column2 = getattr(table, snd_column_name)
        if target_column_name != None:
            if snd_by_name != None:
                result = [i.__dict__[target_column_name] for i in session.query(table).filter(column == by_name, column2 == snd_by_name).all()]
            else:
                result = [i.__dict__[target_column_name] for i in session.query(table).filter(column == by_name).all()]
        else:
            if snd_by_name is not None:
                result = [i.__dict__ for i in session.query(table).filter(column == by_name, column2 == snd_by_name).all()]
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
        result.style.applymap(color_background)
    # elif len(result) != 0:
    #     result = result[0]
    return result

def update_data(table: str,
                by_name: str | list,
                target_column: str,
                edit_column: str,
                new_value: int | str | float | bool,
                isbatch: bool) -> None:
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.new
    table = db_list[table]
    if not isbatch:
        transaction = query_data_object(table, by_name, column_name= target_column )[0][0]
        setattr(transaction, edit_column, new_value)
    else:
        for name in by_name:
            transaction = query_data_object(table, name, target_column)
            setattr(transaction, edit_column, new_value)
    session.commit()

def delete_data(table: str,
                by_primary_key: str | list = None,
                by_name: str = None,
                column_name: str = None,
                isbatch: bool = False,
                ) -> None:
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.new
    table = db_list[table]
    if not isbatch:
        if by_name == None:
            transaction = session.get(table, by_primary_key)
            session.delete(transaction)
        else:
            query = query_data_object(table=table, by_name=by_name, column_name=column_name)
            transaction = [v[0] for v in query]
            for t in transaction:
                session.delete(t)
        
    else:
        for item in by_primary_key:
            transaction = session.get(table, item)
            session.delete(transaction)
    session.commit()

def transcation_rollback():
    from amworkflow.src.infrastructure.database.engine.engine import session
    session.rollback()
    
def query_join_tables(table: str, join_column: str, table1: str,  join_column1:str, table2: str = None, join_column2:str = None, filter0: str = None, filter1: str = None, filter2: str = None, on_column_tb: str = None, on_column_tb1: str = None, on_column_tb2: str = None):
    from amworkflow.src.infrastructure.database.engine.engine import session
    table = db_list[table]
    table1 = db_list[table1]
    on_c0 = getattr(table, join_column)
    on_c1 = getattr(table1, join_column1)
    if on_column_tb is not None:
        column0 = getattr(table, on_column_tb)
    q = session.query(table).join(table1, on_c0 == on_c1)
    c = []
    if filter0 is not None:
        c0 = filter0 == column0
        c.append(c0)
    if filter1 is not None:
        column1 = getattr(table, on_column_tb1)
        c1 = filter1 == column1
        c.append(c1)
    if table2 is not None:
        table2= db_list[table2]
        column2 = getattr(table, on_column_tb2)
        q = q.join(table2)
        if filter2 is not None:
            c2 = filter2 == column2
            c.append(c2)
    if len(c) != 0:
        q.filter(*c)
    result = [i.__dict__ for i in q.all()]
    for dd in result:
        dd.pop("_sa_instance_state", None)
    
    return pd.DataFrame(result)
    
