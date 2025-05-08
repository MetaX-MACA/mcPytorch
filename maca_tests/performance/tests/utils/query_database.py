import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(dir_path + "/../../spd_db/")
from DatabaseManager import DatabaseManager
import argparse

import pandas as pd
import numpy as np
import json
import copy

def get_db176(table_name2='reports'):
    db_config2 = {
        'user': 'root',
        'password': 'metax1234',
        'host': '10.2.120.176',
        'database': 'acl_performance',
        'port': 30000
    }
    db_176 = DatabaseManager(table_name2, db_config2)
    return db_176

def gen_new_json(json_path):
    assert os.path.exists(json_path)

    df = pd.read_json(json_path, encoding="utf-8")
    # print(df.columns)
    # today time
    seconds = df["performance"].values
    seconds_list = []
    for record in seconds:
        seconds_list.append(record['metrics']['second'])
    df['second'] = seconds_list
    df=df.drop(columns="performance")
    # print(len(df))

    # query database BEGIN
    db_176 = get_db176()
    job_name = "pytorch_performance"
    branch = "dev"
    func_names = df['function'].unique()
    # print(func_names)

    # quering database table col names
    '''
    db_col_names = ['id', 'date', 'function', 'testcase', 'testcase_fullname', \
                    'status', 'teststart', 'testend', 'duration', 'testgroup',\
                    'feature', 'hardware', 'job_name', 'software_version', 'branch', \
                    'build_url', 'real_time', 'cpu_time', 'threads', 'iterations', \
                    'metric1', 'metric2', 'metric3', 'metric4', 'metric5', \
                    'golden1', 'golden2', 'golden3', 'golden4', 'golden5', \
                    'arg1', 'arg2', 'arg3', 'arg4', 'arg5', \
                    'arg6', 'arg7', 'arg8', 'arg9', 'arg10',\
                    'n', 'last_10day_avg_second']
    dropped_col_names = ['id', 'date', 'testcase_fullname', \
                    'status', 'teststart', 'testend', 'duration', \
                    'feature', 'hardware', 'job_name', 'software_version', 'branch', \
                    'build_url', 'real_time', 'cpu_time', 'threads', 'iterations', \
                    'metric2', 'metric3', 'metric4', 'metric5', \
                    'golden1', 'golden2', 'golden3', 'golden4', 'golden5', \
                    'arg1', 'arg2', 'arg3', 'arg4', 'arg5', \
                    'arg6', 'arg7', 'arg8', 'arg9', 'arg10',\
                    'n']
    '''
    db_col_names = db_176.query_table_columns_name()
    db_col_names = [str(x).strip('(').strip(')').replace(',', '').strip('\'') for x in db_col_names]
    db_col_names += ['n', 'last_10day_avg_second']
    necessary_cols = ['function', 'testcase', 'testgroup','metric1','last_10day_avg_second']
    dropped_col_names = copy.deepcopy(db_col_names)
    for i in necessary_cols:
        dropped_col_names.remove(i)
    
    # query database by function name, database computes last 10 avg time
    db_data_list = []
    for function in func_names:
        last_data_avg10 = db_176.query_data_groupby_testcase_order_avg10([function, job_name, branch], \
            "function = %s and job_name = %s and branch = %s")
        db_data_list += last_data_avg10

    # list to DataFrame
    db_data = pd.DataFrame(db_data_list, columns=db_col_names)
    db_data = db_data.drop(columns=dropped_col_names)
    db_data.rename(columns={'metric1': 'last_second'}, inplace=True)

    # convert bytearray type to str
    # str_df = db_data.select_dtypes([object])
    # str_df = str_df.stack().str.decode('utf-8').unstack()
    # for col in str_df:
    #     db_data[col] = str_df[col]
    
    # merge
    compare_columns = ['function', 'testcase', 'testgroup']
    out_df = pd.merge(df, db_data, on=compare_columns, how='left')
    # cal time change percentage
    out_df['ref_err'] = (out_df['last_second'] - out_df["second"])/out_df['last_second']
    out_df['ref_err_10'] = (out_df['last_10day_avg_second'] - out_df["second"])/out_df['last_10day_avg_second']

    # print(len(out_df))
    # print(out_df.columns)
    # print(out_df.dtypes)

    # output json
    out_json = out_df.to_json(orient='records')
    data = json.loads(out_json)
    new_json_list = []
    for single_json in data:
        metrics = {"second": single_json['second']}
        metrics2 = {"last_second": single_json['last_second']}
        metrics3 = {"ref_err": single_json['ref_err']}
        metrics4 = {"last_10day_avg_second": single_json['last_10day_avg_second']}
        metrics5 = {"ref_err_10day": single_json['ref_err_10']}

        single_json['performance'] = {"metrics":metrics}
        single_json['performance']["metrics"].update(metrics2)
        single_json['performance']["metrics"].update(metrics3)
        single_json['performance']["metrics"].update(metrics4)
        single_json['performance']["metrics"].update(metrics5)

        single_json.pop('second')
        single_json.pop('last_second')
        single_json.pop('ref_err')
        single_json.pop('last_10day_avg_second')
        single_json.pop('ref_err_10')

        if "softmax" not in single_json['function']:
            single_json.pop('is_log_softmax')

        new_json_list.append(single_json)

    with open(dir_path + "/../../perf_json/" + args.name, "w") as f:
        f.seek(0)
        f.truncate()
        json.dump(new_json_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, metavar='name', help='json file name')
    args = parser.parse_args()

    gen_new_json(dir_path + "/../../perf_json/" + args.name)