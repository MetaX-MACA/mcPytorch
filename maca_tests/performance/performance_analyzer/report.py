from datetime import datetime
from datetime import timedelta
from datetime import time
import re
import logging

import pytz

from peewee import fn
from playhouse.shortcuts import model_to_dict

from database import Report
from database import Target
from rules import get_rule

MAX_METRICS_NUM = 5

class PerformanceReport:
    def __init__(self, case_result):
        self.case_result = case_result
        self.testcase = self.case_result['testcase']
        self.metric_names, self.metrics, self.rule_names = PerformanceReport.parse_metrics(case_result['performance']['metrics'])

    @staticmethod
    def parse_metrics(metrics):
        names = list(metrics.keys())[:MAX_METRICS_NUM]
        values = []
        rule_names = []
        for v in list(metrics.values())[:MAX_METRICS_NUM]:
            if isinstance(v, dict):
                values.append(v['value'])
                rule_names.append(v['rule'])
            else:
                values.append(v)
                rule_names.append(None)

        return names, values, rule_names

    @staticmethod
    def getByIndex(r, prefix, index_list):
        values = []
        for idx in index_list:
            values.append(getattr(r, prefix + '{}'.format(idx + 1)))
        return values

    @staticmethod
    def setByIndex(r, prefix, values):
        for idx, value in enumerate(values, 1):
            setattr(r, prefix + '{}'.format(idx), value)

    def set_args(self, r):
        # TODO
        pass

    def local_time(self, dt_str):
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').astimezone(pytz.utc)

    @staticmethod
    def setOptionalData(r, data):
        if not data:
            return

        for k, v in data.items():
            if hasattr(r, k):
                setattr(r, k, v)

    @staticmethod
    def parseDuration(time_str):
        pattern = re.compile(r'(\d+):(\d+):(\d+)(\.(\d+))?')
        match = pattern.match(time_str)
        if not match:
            raise ValueError('Invalid time string format: ' + time_str)

        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3))
        if match.group(5) is None:
            micro_second = 0
        else:
            # Pad right with 0 to ensure 6 digits
            s = match.group(5).ljust(6, '0')
            if len(s) > 6:
                raise ValueError('Invalid time string format: ' + time_str)
            micro_second = int(s[:6])

        return time(hour, minute, second, micro_second)

    @staticmethod
    def splitString(input_string):
        separators = [',', ';', '|']
        result = []

        current = ""
        for char in input_string:
            if char in separators:
                if current:
                    result.append(current)
                    current = ""
            else:
                current += char

        if current:
            result.append(current)

        return result

    def report(self):
        str_lst = PerformanceReport.splitString(self.case_result['feature'])
        for feature in str_lst:
            r = Report()
            curr_date = self.local_time(self.case_result['teststart']).replace(hour=0, minute=0, second=0, microsecond=0)
            r.date=curr_date
            r.testcase=self.testcase
            r.status = self.case_result['status']
            r.teststart = self.local_time(self.case_result['teststart'])
            r.duration= PerformanceReport.parseDuration(self.case_result['duration'])
            r.testgroup=self.case_result['testgroup']
            #TODO(liuyuxin): feature to be supported
            #r.feature=self.case_result['feature']
            r.hardware = self.case_result['platform']
            r.function = self.case_result['function']
            r.feature = feature
            r.job_name = self.case_result['job_name']
            r.software_version = self.case_result['software_version']
            r.branch = self.case_result['branch']
            PerformanceReport.setByIndex(r, 'metric', self.metrics)
            # maxes, mins, avgs, goldens = PerformanceReport.calculate_golden(self.rule_names, self.local_time(self.case_result['teststart']), self.case_result['job_name'],
            #     self.testcase, self.case_result['platform'], self.case_result['branch'])
            # PerformanceReport.setByIndex(r, 'golden', goldens)
            self.set_args(r)
            PerformanceReport.setOptionalData(r, self.case_result.get('optional'))
            # PerformanceReport.check_set_regression(r, self.rule_names, maxes, mins, avgs, goldens)
            r.save(force_insert=True)
            logging.info(model_to_dict(r))

    @staticmethod
    def calculate_golden(rule_names, end_date, job_name, testcase, hardware, branch):
        query = Report.select(
            Report.testcase,
            fn.MAX(Report.metric1).alias('max1'),
            fn.MIN(Report.metric1).alias('min1'),
            fn.AVG(Report.metric1).alias('avg1'),
            fn.MAX(Report.metric2).alias('max2'),
            fn.MIN(Report.metric2).alias('min2'),
            fn.AVG(Report.metric2).alias('avg2'),
            fn.MAX(Report.metric3).alias('max3'),
            fn.MIN(Report.metric3).alias('min3'),
            fn.AVG(Report.metric3).alias('avg3'),
            fn.MAX(Report.metric4).alias('max4'),
            fn.MIN(Report.metric4).alias('min4'),
            fn.AVG(Report.metric4).alias('avg4'),
            fn.MAX(Report.metric5).alias('max5'),
            fn.MIN(Report.metric5).alias('min5'),
            fn.AVG(Report.metric5).alias('avg5')
            ).where(
                (Report.job_name == job_name) &
                (Report.hardware == hardware) &
                (Report.branch == branch) &
                (Report.testcase == testcase) &
                (Report.status.in_(['Passed', 'Performance Regression'])) &
                (Report.teststart <= end_date)
            ).group_by(Report.testcase)
        logging.info(query.sql())
        results = list(query)
        if results:
            maxes = PerformanceReport.getByIndex(results[0], 'max', range(MAX_METRICS_NUM))
            mins = PerformanceReport.getByIndex(results[0], 'min', range(MAX_METRICS_NUM))
            avgs = PerformanceReport.getByIndex(results[0], 'avg', range(MAX_METRICS_NUM))
            logging.info("max {}, min {}, avg {}".format(maxes, mins, avgs))
            goldens = []
            for rn, m, mi, a in zip(rule_names, maxes, mins, avgs):
                rule = get_rule(rn)
                goldens.append(rule.calculate_golden(m, mi, a))
            return maxes, mins, avgs, goldens
        else:
            logging.info('no golden')
            x =  (None,) * MAX_METRICS_NUM
            return (x,) * 4

    @staticmethod
    def check_set_regression(r, rule_names, maxes, mins, avgs, goldens):
        if r.status != 'Passed':
            return

        metrics = PerformanceReport.getByIndex(r, 'metric', range(MAX_METRICS_NUM))

        for rn, metric, m, mi, a, g in zip(rule_names, metrics, maxes, mins, avgs, goldens):
            rule = get_rule(rn)
            if rule.check_regression(metric, m, mi, a, g):
                r.status = 'Performance Regression'
                return

    def update_metric_names(self):
        t = {}

        t['job_name'] = self.case_result['job_name']
        t['testcase']=self.testcase
        # padding
        metric_names = self.metric_names + (['']*MAX_METRICS_NUM)
        metric_names = metric_names[:MAX_METRICS_NUM]
        for idx, name in enumerate(metric_names, 1):
            t['metric{}_name'.format(idx)] = name

        records = Target.select().where(
            (Target.job_name == t['job_name']) &
            (Target.testcase == t['testcase']))
        if len(records) == 0:
            Target(**t).save(force_insert=True)
        else:
            Target.update(**t).where(
                (Target.job_name == t['job_name']) &
                (Target.testcase == t['testcase'])).execute()
