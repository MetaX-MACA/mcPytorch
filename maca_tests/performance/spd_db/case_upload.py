import os
import math
import json
import datetime
import argparse
from pprint import pprint
from api_client import APIClient
from collections import defaultdict 
from DatabaseManager import DatabaseManager


def get_mean(second):
    return sum(second) / len(second)


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


def get_vbios_version():
    command = "mx-smi --show-version | grep BIOS"
    with os.popen(command, 'r') as file:
        renderdInfo = file.read()
    renderD = renderdInfo.split(': ')
    print(renderD[-1])
    return renderD[-1].strip()


def build_data(data, yestoday_data, today_str, metric_name="images/s", project_name="C500"):
    if float(yestoday_data[20]) != 0:
        percentage = 1 - float(data[20])/float(yestoday_data[20])
    else:
        percentage = 1
    data_spd = {
        "case_item1" : data[2] if isinstance(data[2], str) else data[2].decode("utf-8"),
        "case_item2" : data[3] if isinstance(data[3], str) else data[3].decode("utf-8"),
        "case_item3" : data[9] if isinstance(data[9], str) else data[9].decode("utf-8"),
        "case_item4" : data[10] if isinstance(data[10], str) else data[10].decode("utf-8"),
        "case_item5" : "",
        "extend1" : percentage,
        "extend2" : data[13] if isinstance(data[13], str) else data[13].decode("utf-8"),
        "extend3" : "",
        "extend4" : "",
        "extend5" : "",
        "case_result": "Passed",
        "metrics": [
            {
                "metric_name" : metric_name,
                "metric_value": data[20],
                "metric_target" : yestoday_data[20],
                "golden_comparison_method" : "max",
                "golden_comparison_target" : 1,
            }
        ]
    }
    if (project_name=="C500_FW"):
        data_spd.update({"vbios_version" : get_vbios_version()})
    return data_spd, data[20]


def transter_datas(branch, project_name):
    job_name = "pytorch_performance"
    resnet50_str = "resnet50_AMP_perf_1C"
    bert_str = "bert_amp"
    today = datetime.datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    yesterday = today - datetime.timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    perf_cases = defaultdict(dict)
    perf_cases['config'] = {
        "project_name": project_name,
        "module_name": "acl",
        "sub_module_name": "pytorch",
        "case_execute_date": today_str,
        "maca_version": os.getenv('MACA_VERSION')[7:],
        "code_branch" :branch
    }
    pprint(perf_cases['config'])
    perf_cases['case_result'] = []
    metric_data = {}
    metric_data['config'] = {
        "project_name": project_name,
        "module_name": "acl",
        "sub_module_name": "pytorch",
        "metric_date": today_str,
    }
    metric_data['metrics'] = []
    db_176 = get_db176()
    results = db_176.query_data([today_str, resnet50_str, bert_str, job_name, branch], "teststart LIKE %s '%%' and (function LIKE %s '%%' or function LIKE %s '%%') and job_name = %s and branch = %s")

    for result_item in results:
        today_data = db_176.query_data_order([today_str, job_name, result_item[2], result_item[3], result_item[9], result_item[10]], \
            "teststart LIKE %s '%%' and job_name = %s and function = %s and testcase = %s and testgroup = %s and feature = %s")[0]
        if(result_item[6] != today_data[6]):
            break
        yestoday_data_list = db_176.query_data_order([yesterday_str, job_name, result_item[2], result_item[3], result_item[9], result_item[10]], \
            "teststart LIKE %s '%%' and job_name = %s and function = %s and testcase = %s and testgroup = %s and feature = %s")
        if yestoday_data_list == []:
            yestoday_data = today_data
        else:
            yestoday_data = yestoday_data_list[0]
        if "bert_amp" in result_item[2]:
            data_spd, run_time = build_data(today_data, yestoday_data, today_str, "sequences/s", project_name=project_name)
            perf_cases['case_result'].append(data_spd)
            data_metric = {
                "maca_version": os.getenv('MACA_VERSION')[7:] if project_name == "C500" else os.getenv('MACA_VERSION'),
                "code_branch":branch,
                "metric_group": result_item[2] + "_" + result_item[3],
                "metric_name": 'pytorch-bert-amp-test(train.total_throughput)',
                "metric_value": run_time,
                "metric_unit": 'sequences/s',
                "metric_target_name": "metric_target_name",
                "metric_target_value": 0,
                "metric_description": "Caluculate the mean of all cases for daily pytorch op test time",
            }
            if project_name == "C500_FW":
                data_metric.update({"vbios_version" : get_vbios_version()})
            metric_data['metrics'].append(data_metric)
        else:
            data_spd, run_time = build_data(today_data, yestoday_data, today_str, project_name=project_name)
            perf_cases['case_result'].append(data_spd)
            data_metric = {
                "maca_version": os.getenv('MACA_VERSION')[7:] if project_name == "C500" else os.getenv('MACA_VERSION'),
                "code_branch":branch,
                "metric_group": result_item[2] + "_" + result_item[3],
                "metric_name": 'pytorch-resnet50-test(train.total_ips)',
                "metric_value": run_time,
                "metric_unit": 'images/s',
                "metric_target_name": "metric_target_name",
                "metric_target_value": 0,
                "metric_description": "Caluculate the mean of all cases for daily pytorch op test time",
                "vbios_version" : get_vbios_version(),
            }
            if project_name == "C500_FW":
                data_metric.update({"vbios_version" : get_vbios_version()})
            metric_data['metrics'].append(data_metric)
    if project_name == "C500_FW":
        perf_cases['config']["maca_version"] = os.getenv('MACA_VERSION')
    db_176.close()

    with open(get_json_file(today_str), 'w') as f:
        json.dump(perf_cases, f, indent=4)
    
    with open(get_metric_file(today_str), 'w') as f:
        json.dump(metric_data, f, indent=4)


def upload_file(project_name):
    # case_url= 'http://10.6.26.44:5050/api/gpu/perf/cases/upload'
    if project_name == "C500":
        case_url= 'https://toolkit.metax-internal.com/api/gpu/perf/cases/upload'
    else:
        case_url= 'https://devopsportal.metax-internal.com/api/gpu/perf/cases/upload'
    client = APIClient(case_url)
    today = datetime.datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    client.post_file(get_json_file(today_str))


def upload_metric_file(project_name):
    # case_url = 'http://10.6.26.44:5050/api/gpu/perf/metrics/upload'
    if project_name == "C500":
        case_url= 'https://toolkit.metax-internal.com/api/gpu/perf/metrics/upload'
    else:
        case_url= 'https://devopsportal.metax-internal.com/api/gpu/perf/metrics/upload'
    client = APIClient(case_url)
    today = datetime.datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    client.post_file(get_metric_file(today_str))


def get_json_file(software_version):
    return f'{software_version}_cases_data.json'


def get_metric_file(software_version):
    return f'{software_version}_metric_data.json'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="dev", help="branch")
    parser.add_argument("--project_name", default="C500", help="project_name")
    args = parser.parse_args()
    branch = str(args.branch)
    project_name = str(args.project_name)
    transter_datas(branch, project_name)
    upload_file(project_name)
    upload_metric_file(project_name)
