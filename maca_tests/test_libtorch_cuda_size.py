import os
import mysql.connector
import argparse

class DatabaseManager:
    def __init__(self, table, db_config):
        self.db_config = db_config
        self.table = table
        self.connection = mysql.connector.connect(**self.db_config)
        self.cursor = self.connection.cursor(buffered=True)

    def insert_data(self, data):
        # import pdb
        # pdb.set_trace()
        keys = ', '.join(data.keys())
        values = ', '.join(['%s'] * len(data))
        sql_query = f"INSERT INTO {self.table} ({keys}) VALUES ({values})"
        self.cursor.execute(sql_query, tuple(data.values()))
        self.connection.commit()

    def update_data(self, set_data, condition_data, condition):
        update_clause = ', '.join([f"{key} = %s" for key in set_data.keys()])
        sql_query = f"UPDATE {self.table} SET {update_clause} WHERE {condition}"
        self.cursor.execute(sql_query, tuple(list(set_data.values()) + list(condition_data.values())))
        self.connection.commit()

    def query_data(self, data, condition):
        sql_query = f"SELECT * FROM {self.table} WHERE {condition}"
        if isinstance(data, list):
            self.cursor.execute(sql_query, tuple(data))
        else:
            self.cursor.execute(sql_query, (data,))
        return self.cursor.fetchall()

    def close(self):
        self.connection.close()

class DataBase:
    def __init__(self, branch:str, benchmark_commit_id:str, current_commit_id:str,
                 libtorch_cuda_so:int, table_name='pytorch_lib_size_monitor') -> None:
        db_config = {
            'user': 'root',
            'password': 'metax1234',
            'host': '10.2.177.43',
            'database': 'acl_performance',
            'port': 30000
        }
        self.db = DatabaseManager(table_name, db_config)
        self.branch = branch
        self.benchmark_commit_id = benchmark_commit_id
        self.current_commit_id = current_commit_id
        self.benchmark = self.get_data(self.branch, self.benchmark_commit_id)
        self.current_size = libtorch_cuda_so
        self.first_commit = len(self.benchmark) == 0

    def get_data(self, branch, commit_id):
        return self.db.query_data([branch, commit_id], "branch = %s and commit_id = %s")

    def compare(self, threshold=32*1024*1024):
        if self.first_commit:
            print("first commit, skip check")
            return True
        diff = self.current_size - self.benchmark[0][-1]
        print(f"libtorch_cuda.so size diff: {diff}")
        return diff < threshold

    def submit(self):
        data_not_exist = len(self.get_data(self.branch, self.current_commit_id)) == 0
        if data_not_exist:
            data = {"branch":self.branch, 
                    "commit_id":self.current_commit_id, 
                    "libtorch_cuda_so":self.current_size}
            self.db.insert_data(data)
        else:
            set_data = {"libtorch_cuda_so":self.current_size}
            condition_data = {"branch":self.branch, 
                              "commit_id":self.current_commit_id}
            self.db.update_data(set_data, condition_data, "branch = %s and commit_id = %s")
        
        self.db.close()
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-branch', type=str, metavar='branch', help='code remote branch name,e.g. origin/dev_2.1')
    parser.add_argument('-benchmark_commit_id', type=str, metavar='commit_id', help='commit id,e.g. 650e883a02c34926c81c33b8a08335a2d1b45c86')
    parser.add_argument('-current_commit_id', type=str, metavar='commit_id', help='commit id,e.g. 290e883a02c34926c81c33b8a08335a2d1b45c86')
    parser.add_argument('-libtorch_cuda_so', type=int, metavar='libtorch_cuda_so', help='current libtorch_cuda.so size(byte),e.g. 1601292544')
    args = parser.parse_args()
    status = False

    try:
        db = DataBase(args.branch, args.benchmark_commit_id, args.current_commit_id, args.libtorch_cuda_so)
        ret = db.compare()
        if ret:
            if db.submit():
                status = True
                print("submit data success")
    except:
        print(f"test_libtorch_cuda_size error") 
    if status:
        print("check pass")
        exit(0)
    else: 
        print("check fail")
        exit(1)
    
