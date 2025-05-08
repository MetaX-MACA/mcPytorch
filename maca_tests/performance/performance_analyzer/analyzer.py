from collections import OrderedDict
import json
import logging
import os
import argparse
import traceback

from database import dataBase
from database import Report
from report import PerformanceReport

def loadJsonFile(fn):
    with open(fn) as f:
        return json.load(f, object_hook=OrderedDict)

def initDataBase(db, db_config):
    db.init(db_config['database_name'], user=db_config['user'],
            password=db_config['password'], host=db_config['host'], port=db_config['port'])

def casePassed(case_result):
    return case_result['case_result'] == "Passed"

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    # get script path
    parser.add_argument('-c', type=str, metavar='config', help='config file')
    parser.add_argument('--save_log', type=str, metavar='save_log', help='performance log save directory')
    parser.add_argument('CASE_RESULT_JSON_FILE', type=str)
    args = parser.parse_args()

    config_file = args.c
    tool_dir = os.path.dirname(os.path.realpath(__file__))
    if config_file is None:
        config_file = os.path.join(tool_dir, 'analyzer.json')
    config = loadJsonFile(config_file)
    case_results = loadJsonFile(args.CASE_RESULT_JSON_FILE)
    if type(case_results) != list:
        case_results = [case_results]
    write_flag = 'w'
    if os.path.exists(os.path.join(args.save_log, "perf.log")):
        write_flag = 'a'
    with open(os.path.join(args.save_log, "perf.log"), write_flag) as file:
        for cr in case_results:
            features = PerformanceReport.splitString(cr["feature"])
            for feature in features:
                perf_str = ""
                if cr["is_optim"]:
                    perf_str = feature + cr["function"] + "," + cr["platform"] + "," + cr["testgroup"] +\
                        "," + cr["testcase"] + ",optimized,time: " + str(cr["performance"]['metrics']['second']) +\
                        "\n"
                else:
                    perf_str = feature + cr["function"] + "," + cr["platform"] + "," + cr["testgroup"] +\
                        "," + cr["testcase"] + ",not optimized,time: " + str(cr["performance"]['metrics']['second']) +\
                        "\n"
                file.write(perf_str)
    if os.environ.get('RUN_BENCHMARK_CI') == '1':
        db = dataBase()
        initDataBase(db, config['database'])
        db.connect()
        success_cnt = 0
        for cr in case_results:
            if cr["is_optim"]:
                try:
                    report = PerformanceReport(cr)
                    with db.atomic() as transaction:
                        report.report()
                        report.update_metric_names()
                    success_cnt += 1
                except:
                    traceback.print_exc()
                    logging.error("failed to save performance data:\n{}".format(json.dumps(cr, indent=4)))
        logging.info("save performance data, total {}, succeed {}".format(len(case_results), success_cnt))
        db.close()

if __name__ == '__main__':
    main()
