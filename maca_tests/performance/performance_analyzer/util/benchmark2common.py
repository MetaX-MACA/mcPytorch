import json
import argparse
from collections import OrderedDict
import os
import re

def get_metrics(function, config, benchmark):
    if benchmark is None:
        return OrderedDict()
    testcase_config = tuple(filter(lambda x: x['name'] == function, config['testList']))[0]
    metric_names = testcase_config['metrics']

    metrics = OrderedDict()
    for metric_name in metric_names:
        metrics[metric_name] = benchmark['benchmarks'][0][metric_name]
    
    return metrics

# merge case json and benchmark json result to common json format
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('BENCHMARK_CONFIG_JSON', help='benchmark config json file')
    parser.add_argument('CASE_JSON', help='case json file')
    parser.add_argument('BENCHMARK_JSON', help='benchmark json file')
    parser.add_argument('-o', required=True, help='output json file')
    args = parser.parse_args()

    # read benchmark config json
    with open(args.BENCHMARK_CONFIG_JSON, 'r') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
    # read case json
    with open(args.CASE_JSON, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    # read benchmark json
    if data['case_result'] == 'Passed':
        with open(args.BENCHMARK_JSON, 'r') as f:
            benchmark = json.load(f, object_pairs_hook=OrderedDict)
    else:
        benchmark = None
    
    common_data = OrderedDict()
    common_data['testgroup'] = data['testtype']
    common_data['feature'] = data['testgroup']
    common_data['testcase'] = data['testcase']
    common_data['teststart'] = data['teststart']
    common_data['duration'] = data['case_duration']
    common_data['status'] = data['case_result']
    common_data['software_version'] = os.environ['package_version'] if ('package_version' in os.environ) else None
    if common_data['status'] == 'Before Run':
        common_data['status'] = 'Timeout'

    common_data['platform'] = os.environ['BENCHMARK_TEST_PLATFORM']
    common_data['job_name'] = 'UMD_Performance'
    build_url = os.environ['BUILD_URL']
    if re.search('_release_', build_url):
        common_data['branch'] = 'release'
    elif re.search('_master_', build_url):
        common_data['branch'] = 'master'
    elif re.search('_dev_', build_url):
        common_data['branch'] = 'dev'
    else:
        common_data['branch'] = ''

    function = re.match(r'[0-9a-zA-Z_<> ]+', data['testcase']).group(0)
    common_data['performance'] = OrderedDict()
    common_data['performance']['metrics'] = get_metrics(function, config, benchmark)

    common_data['optional'] = OrderedDict()
    common_data['optional']['build_url'] = build_url
    common_data['optional']['function'] = function
    if benchmark is not None:
        common_data['optional']['real_time'] =benchmark['benchmarks'][0]['real_time']
        common_data['optional']['cpu_time'] =benchmark['benchmarks'][0]['cpu_time']
        common_data['optional']['threads'] =benchmark['benchmarks'][0]['threads']

    with open(args.o, 'w') as f:
        json.dump(common_data, f, indent=4)

if __name__ == '__main__':
    main()
