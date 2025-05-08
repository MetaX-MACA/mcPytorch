import json
import os

file_path = os.path.dirname(os.path.abspath(__file__)) + "/perf_json/"
if os.path.exists(file_path + "total_benckmark.json"):
    os.remove(file_path + "total_benckmark.json")


file_list = os.listdir(file_path)
merged_data = []


for file in file_list:
    if ".json" in file:
        with open(file_path + file, "r") as f:
            data = json.load(f)
            merged_data.append(data)


with open(file_path + "total_benckmark.json", "w") as f:
    json.dump(merged_data, f)