import collections
file_path  = 'log.txt'
useful_lines = []
with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if "p_e_" in line:
            useful_lines.append(line)
useful_lines_set = set(useful_lines)
dict_info_list = []
for line in useful_lines_set:
    try:
        split_items = line.split(',')
        ndim = int(split_items[1])
        narity = int(split_items[2])
        dict_info = collections.OrderedDict()
        dict_info["src_string"] = line
        dict_info["frequency"] = str(useful_lines.count(line))
        dict_info["ndim"] = str(ndim)
        dict_info["narity"] = str(narity)
        dict_info["shape"] = [split_items[3+i] for i in range(ndim)]
        dict_info["output_stride"] = [split_items[3+ndim+i] for i in range(ndim)]
        for i in range(narity):
            dict_info["input{}_stride".format(str(i))] = [split_items[3+(2+i)*ndim+j] for j in range(ndim)]
        dict_info_list.append(dict_info)
    except:
        print(line)
        pass
sorted_dict_info_list = sorted(dict_info_list, key=lambda x : (x["ndim"], x["narity"], x["src_string"]))
current_ndim = ""
current_narity = ""
for item in sorted_dict_info_list:
    ndim = item["ndim"]
    narity = item["narity"]
    if ndim != current_ndim or narity != current_narity:
        print(f"\nbelow lines are lines whose ndim={ndim} and narity={narity}")
        current_ndim = ndim
        current_narity = narity
    print(item)