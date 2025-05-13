import os
from filecmp import dircmp
import glob
import warnings
import pandas as pd
import dill

import torch

from .utils import get_tensor_infos, TensorSummary, \
    add_time_as_suffix, get_min_val, get_max_val

# Some operator names do not need to match.
def escape4specials(item):
    return item.startswith('torch.Tensor.cpu.') or item.startswith('torch.Tensor.cuda.')

def dir_check_helper(dcmp, is_sub=False):
    left_dir = dcmp.left
    right_dir = dcmp.right
    left_only = []
    for item in dcmp.left_only:
        flg = True
        if flg:
            left_only.append(item)
            break    # only one is enough to report error
    right_only = []
    for item in dcmp.right_only:
        flg = True
        if flg:
            right_only.append(item)
            break   # only one is enough to report error
    if left_only or right_only:
        if left_only:
            diff_file = left_only[0]
            diff_dir = left_dir
        else:
            diff_file = right_only[0]
            diff_dir = right_dir
        warnings.warn("Diff file or directory '{}' found in '{}'".format(diff_file, diff_dir))
    msg = "please check or delete irrelevant files or directory."
    if dcmp.common_files and dcmp.common_dirs:
        raise RuntimeError("Both files '{}' and directory '{}' exist in {} and {}, {}".format( \
            dcmp.common_files[0], dcmp.common_dirs[0], left_dir, right_dir, msg))
    elif dcmp.common_dirs and is_sub:
        raise RuntimeError("Sub directory '{}' found in '{}', {}".format( \
            dcmp.common_dirs[0], left_dir, msg))
    elif not dcmp.common_files:
        if not dcmp.common_dirs:
            warnings.warn("Empty directory found in {} and {}, {}".format( \
                left_dir, right_dir, msg))
        return True
    return False

def check_dir(dcmp):
    is_empty = dir_check_helper(dcmp)
    is_sub_empty = True
    for sub_dir in dcmp.common_dirs:
        assert sub_dir.startswith('rank'), "The name of the subdirectory of dir0 and dir1 " \
                                           "must starts with rank."
        is_cur_empty = dir_check_helper(dcmp.subdirs[sub_dir], True)
        is_sub_empty = is_sub_empty and is_cur_empty
    return is_empty and is_sub_empty

@torch.no_grad()
def cal_diff(base_data_in, eval_data_in):
    base_data = base_data_in.float()
    eval_data = eval_data_in.float()
    if base_data.size() != eval_data.size():
        raise ValueError()

    epsilon = 1e-8
    divisor = base_data.abs().sum().item()
    diff1 = (torch.sub(base_data, eval_data).abs().sum() / max(divisor, epsilon)).item()
    divisor = base_data.pow(2).sum().item()
    diff2 = (torch.sub(base_data, eval_data).pow(2).sum() / max(divisor, epsilon)).sqrt().item()

    return diff1, diff2

def comp_dir(dir0, dir1, output_dir, sub_dir = ''):
    res_file = "compare_result_" + sub_dir
    res_file = add_time_as_suffix(res_file)
    left_dir = os.path.join(dir0, sub_dir)
    right_dir = os.path.join(dir1, sub_dir)
    lfs_list = torch.load(os.path.join(left_dir, "files_seqs.pt"))
    rfs_list = torch.load(os.path.join(right_dir, "files_seqs.pt"))
    columns = ["Dev0 Name", "Dev1 Name", "Dev0 Tensor Dtype", "Dev1 Tensor Dtype",
        "Dev0 Tensor Size", "Dev1 Tensor Size", "Dev0 Tensor Max", "Dev1 Tensor Max",
        "Dev0 Tensor Min", "Dev1 Tensor Min", "Diff2 Error", "Diff1 Error",
        "Dev0 Stack", "Dev1 Stack"]
    cmp_res = []
    for lf_name, rf_name in zip(lfs_list, rfs_list):
        lp = os.path.join(left_dir, lf_name)
        rp = os.path.join(right_dir, rf_name)
        if lf_name != rf_name:
            lsplit = lf_name.split('.')
            rsplit = rf_name.split('.')
            lop_api = '.'.join(lsplit[:-4])
            rop_api = '.'.join(rsplit[:-4])
            if not (((escape4specials(lf_name) and escape4specials(rf_name))) \
                and '.'.join(lsplit[-4:]) == '.'.join(rsplit[-4:])):
                raise RuntimeError("Mismatched pt files '{}' and '{}' are found, "
                    "cannot compare two different operator sequences.".format(lp, rp))
        lpt = torch.load(lp, map_location="cpu", pickle_module=dill)
        infos0 = get_tensor_infos()
        rpt = torch.load(rp, map_location="cpu", pickle_module=dill)
        infos1 = get_tensor_infos()
        if len(infos0) != len(infos1):
            raise RuntimeError("The numbers of tensors found in '{}' and '{}' are not equal.". \
                format(lp, rp))
        if len(infos0) > 0:
            is_summary = isinstance(infos0[0], TensorSummary)
        for j, (lt, rt) in enumerate(zip(infos0, infos1)):
            res_item = []
            res_item.append(lf_name[:-2] + str(j))
            res_item.append(rf_name[:-2] + str(j))
            res_item.append(str(lt.dtype))
            res_item.append(str(rt.dtype))
            res_item.append(str(tuple(lt.size())))
            res_item.append(str(tuple(rt.size())))
            if is_summary:
                l_max_val = lt.max_val
                r_max_val = rt.max_val
                l_min_val = lt.min_val
                r_min_val = rt.min_val
                diff1 = diff2 = "NA"
            else:
                l_max_val = get_max_val(lt)
                r_max_val = get_max_val(rt)
                l_min_val = get_min_val(lt)
                r_min_val = get_min_val(rt)
                try:
                    diff1, diff2 = cal_diff(lt, rt)
                except ValueError:
                    msg = "The sizes of compared tensors {} and {} are not equal.".format( \
                        lt.size(), rt.size())
                    warnings.warn("Calc the diff of the values of '{}' and '{}' " \
                        "in '{}' and '{}' failed！ {}".format(res_item[0], res_item[1], lp, rp, msg))
                    diff1 = diff2 = "NA"
            res_item.extend(map(lambda v: str("NA" if v is None else round(v, 4)),
                [l_max_val, r_max_val, l_min_val, r_min_val]))
            res_item.append(diff2 if isinstance(diff2, str) else str(round(diff2, 4)))
            res_item.append(diff1 if isinstance(diff1, str) else str(round(diff1, 4)))
            res_item.append(lpt['stack'] if isinstance(lpt, dict) and 'stack' in lpt.keys() \
                else "NA")
            res_item.append(lpt['stack'] if isinstance(lpt, dict) and 'stack' in rpt.keys() \
                else "NA")
            cmp_res.append(res_item)

    res_df = pd.DataFrame(cmp_res, columns=columns)
    res_df.to_csv(os.path.join(output_dir, res_file), index=False)

def gen_report(dir0, dir1, output_dir="./"):
    assert isinstance(dir0, str) and isinstance(dir1, str), \
        "The types of dir0 and dir1 must be str."
    assert os.path.exists(dir0), "The directory of dir0 is not exist."
    assert os.path.exists(dir1), "The directory of dir1 is not exist."

    dcmp = dircmp(dir0, dir1)
    if check_dir(dcmp):
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if dcmp.common_dirs:
        for sub_dir in dcmp.common_dirs:
            comp_dir(dir0, dir1, output_dir, sub_dir)
    else:
        comp_dir(dir0, dir1, output_dir)

