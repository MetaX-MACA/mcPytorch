import os
import csv

c500_platform = "c500_chip"
a100_platform = "a100"

def perf_path(function, platform):
    return function + "_" + platform + ".log"

def reset_perf_data(function, platform):
    path = perf_path(function, platform)
    if os.path.exists(path):
        os.remove(path)


def record_perf_data(function, platform, test_case, dim0, dim1, time):
    path = perf_path(function, platform)
    print_line = test_case + "|time: " + str(time) +  "|dim0: " + str(dim0) + "|dim1: " + str(dim1) + "\n"
    with open(path, mode="a") as f:
        f.write(print_line)

class PerfRecord:
    def __init__(self, test_case, dim0, dim1, perf_ratio):
        self.test_case = test_case
        self.dim0 = dim0
        self.dim1 = dim1
        self.perf_ratio = perf_ratio

class PerfTable:
    def __init__(self, test_case):
        self.test_case = test_case
        self.dim0_list = []
        self.dim1_list = []
        self.shape_list = []
        self.perf_list = []
    
    def update(self, record: PerfRecord):
        self.shape_list.append([record.dim0, record.dim1])
        if record.dim0 not in self.dim0_list:    
            self.dim0_list.append(record.dim0)
        if record.dim1 not in self.dim1_list:
            self.dim1_list.append(record.dim1)
        self.perf_list.append(record.perf_ratio)
    
    def to_csv_rows(self):
        csv_rows = []
        csv_rows.append([self.test_case])
        if len(self.dim0_list) * len(self.dim1_list) == len(self.perf_list):
            cur_csv_row = self.dim1_list.copy()
            cur_csv_row.insert(0, " ")
            csv_rows.append(cur_csv_row)
            for i, dim0 in enumerate(self.dim0_list):
                cur_csv_row = [dim0]
                cur_csv_row += self.perf_list[(i * len(self.dim1_list)):((i+1) * len(self.dim1_list))]
                csv_rows.append(cur_csv_row)
        else:
            csv_rows.append(self.shape_list)
            csv_rows.append(self.perf_list)
        
        return csv_rows


def parse_perf_data(c500_line, a100_line):
    test_case = c500_line.split("|time:")[0].strip()
    c500_time = float(c500_line.split("|time:")[1].split("|dim0: ")[0].strip())
    a100_time = float(a100_line.split("|time:")[1].split("|dim0: ")[0].strip())
    dim0 = c500_line.split("|dim0:")[1].split("|dim1: ")[0].strip()
    dim1 = c500_line.split("|dim1:")[1].strip()
    return PerfRecord(test_case, dim0, dim1, c500_time/a100_time)

from numpy import mean

def total_analyse(all_tables: dict):
    record_size = 0
    less_than_2_size = 0
    less_than_3_size = 0
    sum_ratio = 0
    over_10_size = 0
    all_results = []
    for test_case in all_tables.keys():
        table: PerfTable = all_tables[test_case]
        mean_ratio = mean(table.perf_list)
        count = len(table.perf_list)
        record_size += count
        sum_ratio += sum(table.perf_list)
        less_than_2_count = len([i for i  in table.perf_list if i < 2])
        less_than_3_count = len([i for i  in table.perf_list if i < 3])
        over_10_count = len([i for i  in table.perf_list if i > 10])
        less_than_2_size += less_than_2_count
        less_than_3_size += less_than_3_count
        over_10_size += over_10_count
        all_results.append({"test_case": test_case, "mean_ratio": mean_ratio, "less_than_2_ratio": less_than_2_count / count, "less_than_3_ratio": less_than_3_count / count, "over_10_count": over_10_count})
    all_results.sort(key=lambda item: item["mean_ratio"], reverse=True)
    for result in all_results:
        print(result)
    print("testcase size: {}, mean_ratio: {}, less_than_2_ratio: {}, less_than_3_ratio: {}, over_10_size: {}".format(record_size, sum_ratio / record_size, less_than_2_size / record_size, less_than_3_size / record_size, over_10_size))

def statistic_perf_data(function_name):
    c500_path = perf_path(function_name, c500_platform)
    a100_path = perf_path(function_name, a100_platform)
    if not os.path.exists(c500_path) or not os.path.exists(a100_path):
        print("c500 or a100 perf data is not ready for {}.".format(function_name))
        return
    csv_path = function_name + "_perf_overview.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
    c500_lines = []
    with open(c500_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line != '':
                c500_lines.append(line)

    a100_lines = []
    with open(a100_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line != '':
                a100_lines.append(line)
    print("c500_lines: ", len(c500_lines))
    print("a100_lines: ", len(a100_lines))
    assert len(c500_lines) == len(a100_lines)

    all_record = {}
    for i in range(len(c500_lines)):
        perf_record = parse_perf_data(c500_lines[i], a100_lines[i])
        print(perf_record.test_case, perf_record.dim0, perf_record.dim1, perf_record.perf_ratio)
        if perf_record.test_case not in all_record.keys():
            perf_table = PerfTable(perf_record.test_case)
            all_record[perf_record.test_case] = perf_table
        all_record[perf_record.test_case].update(perf_record)
    
    total_analyse(all_record)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        for perf_table in all_record.values():
            csv_writer.writerows(perf_table.to_csv_rows())

