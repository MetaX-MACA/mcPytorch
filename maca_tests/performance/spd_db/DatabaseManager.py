import mysql.connector
from multiprocessing import Pool

import time
from datetime import datetime

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class DatabaseManager:
    def __init__(self, table, db_config):
        self.db_config = db_config
        self.table = table
        self.processes_num = 20
        self.connection = mysql.connector.connect(**self.db_config)
        self.cursor = self.connection.cursor(buffered=True)

    @timer_decorator
    def insert_data(self, data):
        keys = ', '.join(data.keys())
        values = ', '.join(['%s'] * len(data))
        sql_query = f"INSERT INTO {self.table} ({keys}) VALUES ({values})"
        self.cursor.execute(sql_query, tuple(data.values()))
        self.connection.commit()

    def update_data(self, data, condition):
        update_clause = ', '.join([f"{key} = %s" for key in data.keys()])
        sql_query = f"UPDATE {self.table} SET {update_clause} WHERE {condition}"
        self.cursor.execute(sql_query, tuple(data.values()))
        self.connection.commit()

    def query_data(self, data, condition):
        sql_query = f"SELECT * FROM {self.table} WHERE {condition} ORDER BY teststart DESC"
        if isinstance(data, list):
            self.cursor.execute(sql_query, tuple(data))
        else:
            self.cursor.execute(sql_query, (data,))
        return self.cursor.fetchall()
    
    def query_data_order(self, data, condition):
        sql_query = f"SELECT * FROM {self.table} WHERE {condition} ORDER BY teststart DESC LIMIT 1"
        if isinstance(data, list):
            self.cursor.execute(sql_query, tuple(data))
        else:
            self.cursor.execute(sql_query, (data,))
        return self.cursor.fetchall()
    
    def query_data_order_10(self, data, condition):
        sql_query = f"SELECT * FROM {self.table} WHERE {condition} ORDER BY teststart DESC LIMIT 10"
        if isinstance(data, list):
            self.cursor.execute(sql_query, tuple(data))
        else:
            self.cursor.execute(sql_query, (data,))
        return self.cursor.fetchall()

    def query_data_groupby_testcase_order(self, data, condition):
        sql_query = f"\
            SELECT * \
            FROM (\
                SELECT *, row_number() OVER (PARTITION BY testgroup, testcase ORDER BY teststart DESC) AS n \
                FROM {self.table}\
                WHERE {condition}\
            ) AS x \
            WHERE n <= 1"
        if isinstance(data, list):
            self.cursor.execute(sql_query, tuple(data))
        else:
            self.cursor.execute(sql_query, (data,))
        return self.cursor.fetchall()

    def query_data_groupby_testcase_order_avg10(self, data, condition):
        sql_query = f"\
            SELECT *, AVG(metric1) as time_avg10\
            FROM (\
                SELECT *, row_number() OVER (PARTITION BY testgroup, testcase ORDER BY teststart DESC) AS n \
                FROM {self.table}\
                WHERE {condition}\
            ) AS x \
            WHERE n <= 10\
            GROUP BY testgroup, testcase\
            "
        if isinstance(data, list):
            self.cursor.execute(sql_query, tuple(data))
        else:
            self.cursor.execute(sql_query, (data,))
        return self.cursor.fetchall()

    def query(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def query_table_columns_name(self):
        query = f"SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA='{self.db_config['database']}' AND TABLE_NAME ='{self.table}'"
        self.cursor.execute(query)
        return self.cursor.fetchall()
        
    def get_unique_case_execute_dates(self):
        #only for 51 PerfCase
        sql_query = "SELECT DISTINCT case_execute_date FROM PerfCase ORDER BY case_execute_date DESC"
        self.cursor.execute(sql_query)
        return self.cursor.fetchall()

    @timer_decorator
    def batch_insert(self, data_list):
        with Pool(processes=self.processes_num) as pool:
            pool.map(self.insert, data_list)

    @timer_decorator
    def batch_update(self, data_list):
        with Pool(processes=self.processes_num) as pool:
            pool.map(self.update, data_list)

    def close(self):
        self.connection.close()
