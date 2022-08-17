import copy
import os
import numpy as np

from config import case_name

max_cpi = 99.99
max_power = 99.99

def CPI_metric(x: np.ndarray):
    version_to_match = var_to_version(x)
    metric = 99.99  # max
    for case in metrics_all:
        if case['version'] == version_to_match:
            metric = case['CPI']
            break
    # print(f"call CPIMetric x: {x}, y: {metric}, version: {version_to_match}")
    if 99.98 < metric:
        print(f"missing: call CPIMetric x: {x}, version: {version_to_match}")
    elif metric < 0.2:
        print(f"missing: call CPIMetric x: {x}, version: {version_to_match}")
    return metric


def power_metric(x: np.ndarray):
    version_to_match = var_to_version(x)
    metric = 99.99  # max
    for case in metrics_all:
        if case['version'] == version_to_match:
            metric = case['Power']
            break
    # print(f"call PowerMetric x: {x}, y: {metric}, version: {version_to_match}")
    if 99.98 < metric:
        print(f"missing: call PowerMetric x: {x}, version: {version_to_match}")
    elif metric < 0.2:
        print(f"missing: call PowerMetric x: {x}, version: {version_to_match}")
    return metric


def get_ref_point():
    return [max_cpi + 1, max_power + 1]


def get_max_hv():
    return (max_cpi + 1) * (max_power + 1)


SIMULATOR_CYCLES_PER_SECOND_map = {
    '0': 1000000000,
    '1': 1500000000,
    '2': 2000000000,  # 2GHz
    '3': 3000000000,
}

IFQ_SIZE_map = {
    '0': 8,
    '1': 16,
}

DECODEQ_SIZE_map = {
    '0': 8,
    '1': 16,
}
FETCH_WIDTH_map = {
    '0': 2,
    '1': 4,
    '2': 8,
    '3': 16,
}
DECODE_WIDTH_map = {
    '0': 2,
    '1': 3,
    '2': 4,
    '3': 5,
}

DISPATCH_WIDTH_map = {
    '0': 2,
    '1': 4,
    '2': 5,
    '3': 6,
}

COMMIT_WIDTH_map = {
    '0': 2,
    '1': 4,
    '2': 6,
    '3': 8,
}

PHY_GPR_NUM_map = {
    '0': 40,
    '1': 64,
    '2': 128,
    '3': 180,
}

PHY_FGPR_NUM_map = {
    '0': 40,
    '1': 64,
    '2': 128,
    '3': 180,
}

GPR_WRITEBACK_WIDTH_map = {
    '0': 2,
    '1': 4,
}

FGPR_WRITEBACK_WIDTH_map = {
    '0': 2,
    '1': 4,
}

RUU_SIZE_MAX_map = {
    '0': 32,
    '1': 64,
    '2': 128,
    '3': 256,
}

INT_BP_map = {
    '0': 1,
    '1': 2,
}

INT_ALU_map = {
    '0': 1,
    '1': 2,
}

INT_MULT_map = {
    '0': 1,
    '1': 2,
}

INT_MULT_OP_LAT_map = {
    '0': 2,
    '1': 4
}

INT_MULT_ISSUE_LAT_map = {
    '0': 4,
    '1': 1,
}

INT_DIV_OP_LAT_map = {
    '0': 8,
    '1': 16,
}

INT_DIV_ISSUE_LAT_map = {
    '0': 16,
    '1': 1,
}

FP_ALU_map = {
    '0': 1,
    '1': 2,
}

FP_ALU_MULT_map = {
    '0': 1,
    '1': 2,
}

FP_MULT_DIV_map = {
    '0': 1,
    '1': 2,
}

FP_ALU_MULT_DIV_map = {
    '0': 0,
    '1': 1,
}

FP_MULT_OP_LAT_map = {
    '0': 2,
    '1': 4,
}

FP_MULT_ISSUE_LAT_map = {
    '0': 4,
    '1': 1,
}

FP_DIV_OP_LAT_map = {
    '0': 8,
    '1': 16,
}

FP_DIV_ISSUE_LAT_map = {
    '0': 16,
    '1': 1,
}
'''
FP_SQRT_OP_LAT_map = {
'0' : 4,
'1' : 1,
}

FP_SQRT_ISSUE_LAT_map = {
'0' : 4,
'1' : 1,
}
'''

LOAD_PORT_WIDTH_map = {
    '0': 1,
    '1': 2,
}

STORE_PORT_WIDTH_map = {
    '0': 1,
    '1': 2,
}

LOAD_STORE_PORT_WIDTH_map = {
    '0': 0,
    '1': 2,
}

LOAD_QUEUE_SIZE_map = {
    '0': 10,
    '1': 30,
    '2': 60,
    '3': 90,
}

STORE_QUEUE_SIZE_map = {
    '0': 10,
    '1': 30,
    '2': 60,
    '3': 90,
}

BPRED_map = {
    '0': 'gShare',
    '1': 'tage'
}

RAS_SIZE_map = {
    '0': 8,
    '1': 16,
}

L1_ICACHE_SET_map = {
    '0': 64,
    '1': 128,
    '2': 256,
}

L1_ICACHE_ASSOC_map = {
    '0': 2,
    '1': 4,
    '2': 8,
}

L1_DCACHE_SET_map = {
    '0': 64,
    '1': 128,
    '2': 256,
}

L1_DCACHE_ASSOC_map = {
    '0': 2,
    '1': 4,
    '2': 8,
}

L1_DCACHE_WRITEBACK_map = {
    '0': 0,
    '1': 1
}

L2_CACHE_SET_map = {
    '0': 128,
    '1': 1024,
}

L2_CACHE_ASSOC_map = {
    '0': 4,
    '1': 8,
}

LLC_map = {
    '0': 2,
    # '1' : 3,
}

version_map_id = {
    'IFQ_SIZE': [0, len(IFQ_SIZE_map)],
    'DECODEQ_SIZE': [1, len(DECODEQ_SIZE_map)],
    'FETCH_WIDTH': [2, len(FETCH_WIDTH_map)],
    'DECODE_WIDTH': [3, len(DECODE_WIDTH_map)],
    'DISPATCH_WIDTH': [4, len(DISPATCH_WIDTH_map)],
    'COMMIT_WIDTH': [5, len(COMMIT_WIDTH_map)],
    'PHY_GPR_NUM': [6, len(PHY_GPR_NUM_map)],
    'PHY_FGPR_NUM': [7, len(PHY_FGPR_NUM_map)],
    'GPR_WRITEBACK_WIDTH': [8, len(GPR_WRITEBACK_WIDTH_map)],
    'FGPR_WRITEBACK_WIDTH': [9, len(FGPR_WRITEBACK_WIDTH_map)],
    'RUU_SIZE_MAX': [10, len(RUU_SIZE_MAX_map)],
    'INT_BP': [11, len(INT_BP_map)],
    'INT_ALU': [12, len(INT_ALU_map)],
    'INT_MULT': [13, len(INT_MULT_map)],
    'INT_MULT_OP_LAT': [14, len(INT_MULT_OP_LAT_map)],
    'INT_MULT_ISSUE_LAT': [15, len(INT_MULT_ISSUE_LAT_map)],
    'INT_DIV_OP_LAT': [16, len(INT_DIV_OP_LAT_map)],
    'INT_DIV_ISSUE_LAT': [17, len(INT_DIV_ISSUE_LAT_map)],
    'FP_ALU': [18, len(FP_ALU_map)],
    'FP_ALU_MULT': [19, len(FP_ALU_MULT_map)],
    'FP_MULT_DIV': [20, len(FP_MULT_DIV_map)],
    'FP_ALU_MULT_DIV': [21, len(FP_ALU_MULT_DIV_map)],
    'FP_MULT_OP_LAT': [22, len(FP_MULT_OP_LAT_map)],
    'FP_MULT_ISSUE_LAT': [23, len(FP_MULT_ISSUE_LAT_map)],
    'FP_DIV_OP_LAT': [24, len(FP_DIV_OP_LAT_map)],
    'FP_DIV_ISSUE_LAT': [25, len(FP_DIV_ISSUE_LAT_map)],
    # 'FP_SQRT_OP_LAT': 25,
    # 'FP_SQRT_ISSUE_LAT': 26,
    'LOAD_PORT_WIDTH': [26, len(LOAD_PORT_WIDTH_map)],
    'STORE_PORT_WIDTH': [27, len(STORE_PORT_WIDTH_map)],
    'LOAD_STORE_PORT_WIDTH': [28, len(LOAD_STORE_PORT_WIDTH_map)],
    'LOAD_QUEUE_SIZE': [29, len(LOAD_QUEUE_SIZE_map)],
    'STORE_QUEUE_SIZE': [30, len(STORE_QUEUE_SIZE_map)],
    'BPRED': [31, len(BPRED_map)],
    'RAS_SIZE': [32, len(RAS_SIZE_map)],
    'L1_ICACHE_SET': [33, len(L1_ICACHE_SET_map)],
    'L1_ICACHE_ASSOC': [34, len(L1_ICACHE_ASSOC_map)],
    'L1_DCACHE_SET': [35, len(L1_DCACHE_SET_map)],
    'L1_DCACHE_ASSOC': [36, len(L1_DCACHE_ASSOC_map)],
    'L1_DCACHE_WRITEBACK': [37, len(L1_DCACHE_WRITEBACK_map)],
    'L2_CACHE_SET': [38, len(L2_CACHE_SET_map)],
    'L2_CACHE_ASSOC': [39, len(L2_CACHE_ASSOC_map)],
    'LLC': [40, len(LLC_map)],
    'max': 41,
}

DEF_FREQ = 0
DEF_IFQ = 1
DEF_DECODEQ = 2
DEF_FETCH_WIDTH = 3
DEF_DECODE_WIDTH = 4
DEF_DISPATCH_WIDTH = 5
DEF_COMMIT_WIDHT = 6
DEF_GPR = 7
DEF_FGPR = 8
DEF_GPR_WRITEBACK = 9
DEF_FGPR_WRITEBACK = 10
DEF_RUU_SIZE_MAX = 11
DEF_INT_BP = 12
DEF_FP_ALU = 19
DEF_LOAD_PORT_WIDTH = 29

DEF_MULTI_BTB = 35
DEF_BPRED = 36
DEF_L0_ICACHE = 38
DEF_EXECUTE_RECOVER = 39
DEF_RAW_LOAD_PRED = 40
DEF_PREFETCH_INST = 41
DEF_PREFETCH_DATA = 42
DEF_L1_ICACHE_SET = 43
DEF_L1_DCACHE_SET = 45
DEF_L2_CACHE_SET = 48


# 49

def gen_version_choose(DISPATCH_WIDTH_index, exe_int, exe_fp, lsq, dcache, icache, bp, l2cache):
    version = ['0' for i in range(int(version_map_id['max']))]
    # DISPATCH_WIDTH = DISPATCH_WIDTH_map[version[version_map_id['DISPATCH_WIDTH'][0]]]
    version[version_map_id['DISPATCH_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['IFQ_SIZE'][0]] = str(int(DISPATCH_WIDTH_index / 2))
    version[version_map_id['DECODEQ_SIZE'][0]] = str(int(DISPATCH_WIDTH_index / 2))
    version[version_map_id['FETCH_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['DECODE_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['COMMIT_WIDTH'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['PHY_GPR_NUM'][0]] = str(DISPATCH_WIDTH_index)
    version[version_map_id['PHY_FGPR_NUM'][0]] = str(DISPATCH_WIDTH_index)

    version[version_map_id['GPR_WRITEBACK_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
    version[version_map_id['FGPR_WRITEBACK_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
    version[version_map_id['RUU_SIZE_MAX'][0]] = str(DISPATCH_WIDTH_index)

    if -1 < exe_int:
        version[version_map_id['INT_BP'][0]] = str(exe_int)
        version[version_map_id['INT_ALU'][0]] = str(exe_int)
        version[version_map_id['INT_MULT'][0]] = str(exe_int)
        version[version_map_id['INT_MULT_OP_LAT'][0]] = str(exe_int)
        version[version_map_id['INT_MULT_ISSUE_LAT'][0]] = str(exe_int)
        version[version_map_id['INT_DIV_OP_LAT'][0]] = str(exe_int)
        version[version_map_id['INT_DIV_ISSUE_LAT'][0]] = str(exe_int)
    else:
        version[version_map_id['INT_BP'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_ALU'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_MULT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_MULT_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_MULT_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_DIV_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['INT_DIV_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))

    if -1 < exe_fp:
        version[version_map_id['FP_ALU'][0]] = str(exe_fp)
        version[version_map_id['FP_ALU_MULT'][0]] = str(exe_fp)
        version[version_map_id['FP_MULT_DIV'][0]] = str(exe_fp)
        version[version_map_id['FP_ALU_MULT_DIV'][0]] = str(exe_fp)
        version[version_map_id['FP_MULT_OP_LAT'][0]] = str(exe_fp)
        version[version_map_id['FP_MULT_ISSUE_LAT'][0]] = str(exe_fp)
        version[version_map_id['FP_DIV_OP_LAT'][0]] = str(exe_fp)
        version[version_map_id['FP_DIV_ISSUE_LAT'][0]] = str(exe_fp)
    else:
        version[version_map_id['FP_ALU'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_ALU_MULT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_MULT_DIV'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_ALU_MULT_DIV'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_MULT_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_MULT_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_DIV_OP_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['FP_DIV_ISSUE_LAT'][0]] = str(int(DISPATCH_WIDTH_index / 2))

    if -1 < lsq:
        version[version_map_id['LOAD_PORT_WIDTH'][0]] = str(int(lsq / 2))
        version[version_map_id['STORE_PORT_WIDTH'][0]] = str(int(lsq / 2))
        version[version_map_id['LOAD_STORE_PORT_WIDTH'][0]] = str(int(lsq / 2))
        version[version_map_id['LOAD_QUEUE_SIZE'][0]] = str(lsq)
        version[version_map_id['STORE_QUEUE_SIZE'][0]] = str(lsq)
    else:
        version[version_map_id['LOAD_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['STORE_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['LOAD_STORE_PORT_WIDTH'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['LOAD_QUEUE_SIZE'][0]] = str(DISPATCH_WIDTH_index)
        version[version_map_id['STORE_QUEUE_SIZE'][0]] = str(DISPATCH_WIDTH_index)

    if -1 < bp:
        version[version_map_id['BPRED'][0]] = str(bp)
        version[version_map_id['RAS_SIZE'][0]] = str(bp)
    else:
        version[version_map_id['BPRED'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['RAS_SIZE'][0]] = str(int(DISPATCH_WIDTH_index / 2))

    if -1 < icache:
        version[version_map_id['L1_ICACHE_SET'][0]] = str(icache)
        version[version_map_id['L1_ICACHE_ASSOC'][0]] = str(icache)
    else:
        version[version_map_id['L1_ICACHE_SET'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
        version[version_map_id['L1_ICACHE_ASSOC'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))

    if -1 < dcache:
        version[version_map_id['L1_DCACHE_SET'][0]] = str(dcache)
        version[version_map_id['L1_DCACHE_ASSOC'][0]] = str(dcache)
        version[version_map_id['L1_DCACHE_WRITEBACK'][0]] = str(int(dcache / 2))
    else:
        version[version_map_id['L1_DCACHE_SET'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
        version[version_map_id['L1_DCACHE_ASSOC'][0]] = str(int((1 + DISPATCH_WIDTH_index) / 2))
        version[version_map_id['L1_DCACHE_WRITEBACK'][0]] = str(int((DISPATCH_WIDTH_index) / 2))

    if -1 < l2cache:
        version[version_map_id['L2_CACHE_SET'][0]] = str(l2cache)
        version[version_map_id['L2_CACHE_ASSOC'][0]] = str(l2cache)
    else:
        version[version_map_id['L2_CACHE_SET'][0]] = str(int(DISPATCH_WIDTH_index / 2))
        version[version_map_id['L2_CACHE_ASSOC'][0]] = str(int(DISPATCH_WIDTH_index / 2))

    version[version_map_id['LLC'][0]] = '0'

    version_str = ''
    for version_iter in version:
        version_str += version_iter
    return version_str


var_list = [
    "DispatchWidth"
    , "ExeInt"
    , "ExeFP"
    , "LSQ"
    , "Dcache"
    , "Icache"
    , "BP"
    , "L2cache"
]

var_list_index = [
    version_map_id['DISPATCH_WIDTH'][0],
    version_map_id['INT_ALU'][0],
    version_map_id['FP_ALU'][0],
    version_map_id['LOAD_QUEUE_SIZE'][0],
    version_map_id['L1_DCACHE_SET'][0],
    version_map_id['L1_ICACHE_SET'][0],
    version_map_id['BPRED'][0],
    version_map_id['L2_CACHE_SET'][0],
]

range_list = [
    len(DISPATCH_WIDTH_map)
    , len(INT_ALU_map)
    , len(FP_ALU_map)
    , len(LOAD_PORT_WIDTH_map)
    , len(L1_DCACHE_SET_map)
    , len(L1_ICACHE_SET_map)
    , len(BPRED_map)
    , len(L2_CACHE_SET_map)
]


def get_orthogonal_array():
    samples = []

    samples.append([0, 0, 0, 0, 0, 0, 0, 0])
    samples.append([0, 1, 1, 1, 1, 1, 1, 1])
    samples.append([0, 0, 0, 2, 2, 2, 0, 0])
    samples.append([0, 1, 1, 3, 0, 1, 1, 1])
    samples.append([0, 0, 1, 0, 1, 2, 0, 1])
    samples.append([0, 1, 0, 1, 2, 0, 1, 0])

    samples.append([1, 0, 0, 0, 0, 2, 0, 0])
    samples.append([1, 1, 1, 1, 1, 0, 1, 1])
    samples.append([1, 0, 0, 2, 2, 1, 0, 0])
    samples.append([1, 1, 1, 3, 0, 2, 1, 1])
    samples.append([1, 0, 1, 2, 1, 1, 0, 1])
    samples.append([1, 1, 0, 3, 2, 0, 1, 0])

    samples.append([2, 0, 0, 0, 0, 0, 1, 1])
    samples.append([2, 1, 1, 1, 1, 1, 0, 0])
    samples.append([2, 0, 1, 2, 2, 2, 1, 1])
    samples.append([2, 1, 0, 3, 0, 2, 0, 0])
    samples.append([2, 0, 1, 0, 1, 0, 1, 0])
    samples.append([2, 1, 0, 1, 2, 1, 0, 1])

    samples.append([3, 0, 0, 0, 0, 0, 1, 1])
    samples.append([3, 1, 1, 1, 1, 1, 0, 0])
    samples.append([3, 0, 1, 2, 2, 2, 1, 1])
    samples.append([3, 1, 0, 3, 0, 1, 0, 0])
    samples.append([3, 0, 1, 2, 1, 2, 1, 0])
    samples.append([3, 1, 0, 3, 2, 0, 0, 1])

    samples.append([0, 1, 0, 1, 2, 0, 0, 1])  # random add one

    return samples


def get_myinit_array():
    samples = []

    samples.append([0, 0, 0, 0, 0, 0, 0, 0])
    samples.append([0, 1, 1, 1, 1, 1, 1, 1])
    samples.append([0, 0, 0, 2, 2, 2, 0, 0])
    samples.append([0, 1, 1, 3, 0, 1, 1, 1])
    samples.append([0, 0, 1, 0, 1, 2, 0, 1])
    samples.append([0, 1, 0, 1, 2, 0, 1, 0])

    samples.append([1, 0, 0, 0, 0, 2, 0, 0])
    samples.append([1, 1, 1, 1, 1, 0, 1, 1])
    samples.append([1, 0, 0, 2, 2, 1, 0, 0])
    samples.append([1, 1, 1, 3, 0, 2, 1, 1])
    samples.append([1, 0, 1, 2, 1, 1, 0, 1])
    samples.append([1, 1, 0, 3, 2, 0, 1, 0])

    samples.append([2, 0, 0, 0, 0, 0, 1, 1])
    samples.append([2, 1, 1, 1, 1, 1, 0, 0])
    samples.append([2, 0, 1, 2, 2, 2, 1, 1])
    samples.append([2, 1, 0, 3, 0, 2, 0, 0])
    samples.append([2, 0, 1, 0, 1, 0, 1, 0])
    samples.append([2, 1, 0, 1, 2, 1, 0, 1])

    samples.append([3, 0, 0, 0, 0, 0, 1, 1])
    samples.append([3, 1, 1, 1, 1, 1, 0, 0])
    samples.append([3, 0, 1, 2, 2, 2, 1, 1])
    samples.append([3, 1, 0, 3, 0, 1, 0, 0])
    samples.append([3, 0, 1, 2, 1, 2, 1, 0])
    samples.append([3, 1, 0, 3, 2, 0, 0, 1])

    samples.append([0, 1, 0, 1, 2, 0, 0, 1])  # random add one

    return samples


# for ax SearchSpace
def get_search_space():
    parameters_vec = []
    for index in range(len(var_list)):
        # parameters_vec.append(ChoiceParameter(name=var_list[index], parameter_type=ParameterType.STRING, values=[str(i) for i in range(range_list[index])]))
        parameters_vec.append(RangeParameter(name=var_list[index], lower=0, upper=range_list[index] - 1,
                                             parameter_type=ParameterType.INT))

    search_space = SearchSpace(
        parameters=parameters_vec,
    )
    return search_space


def get_var_list():
    return var_list
    # return ["x"+str(i) for i in range(1, 1 + length(var_list)]


def get_var_list_index():
    return var_list_index


def var_to_version(x: np.ndarray):
    version = gen_version_choose(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
    return version


# Euclidean distance
def distance_f(real_pareto_optimal, learned_pareto_optimal):
    # print(f"real_pareto_optimal={real_pareto_optimal}")
    # print(f"learned_pareto_optimal={learned_pareto_optimal}")
    distance = np.sqrt(np.sum(np.square(np.array(real_pareto_optimal) - np.array(learned_pareto_optimal))))
    return distance


def evaluate_ADRS(real_pareto_optimal_sets, learned_pareto_optimal_sets, coverage=False):
    ADRS = 0
    coverage_num = 0
    # print("real=", real_pareto_optimal_sets)
    # print("leared=", learned_pareto_optimal_sets)
    for real_pareto_optimal in real_pareto_optimal_sets:
        distances = []
        for learned_pareto_optimal in learned_pareto_optimal_sets:
            distances.append(distance_f(real_pareto_optimal, learned_pareto_optimal))
        min_distance = min(distances)
        ADRS += min_distance
        if coverage:
            if min_distance < 0.001:
                coverage_num += 1
    ADRS /= len(real_pareto_optimal_sets)
    coverage_percent = coverage_num / len(real_pareto_optimal_sets)
    if coverage:
        return ADRS, coverage_percent
    else:
        return ADRS


def get_di(learned_pareto_optimal_sets_y_unsort):
    di = []
    learned_pareto_optimal_sets_y_sort = copy.deepcopy(learned_pareto_optimal_sets_y_unsort)
    learned_pareto_optimal_sets_y_sort = np.asarray(sorted(learned_pareto_optimal_sets_y_sort, key=(lambda x: [x[0]])))
    learned_pareto_optimal_sets_y_sort_trans = np.zeros(np.shape(learned_pareto_optimal_sets_y_sort)[0:2])
    learned_pareto_optimal_sets_y_sort_trans[:, 0] = learned_pareto_optimal_sets_y_sort[:, 0] / max(learned_pareto_optimal_sets_y_sort[:, 0])
    learned_pareto_optimal_sets_y_sort_trans[:, 1] = learned_pareto_optimal_sets_y_sort[:, 1] / max(learned_pareto_optimal_sets_y_sort[:, 1])
    #learned_pareto_optimal_sets_y_sort_trans[:, 2] = learned_pareto_optimal_sets_y_sort[:, 2]
    for i in range(len(learned_pareto_optimal_sets_y_sort) - 1):
        di.append(distance_f(learned_pareto_optimal_sets_y_sort_trans[i, 0:2], learned_pareto_optimal_sets_y_sort_trans[i + 1, 0:2]))
    return di, learned_pareto_optimal_sets_y_sort


def evaluate_non_uniformity(learned_pareto_optimal_sets_y_unsort):
    di, _ = get_di(learned_pareto_optimal_sets_y_unsort)
    non_uniformity = sum(np.abs(di - np.mean(di))) / (np.sqrt(2) * (len(learned_pareto_optimal_sets_y_unsort) - 1))
    return non_uniformity


def read_metrics(config_dir, case_name):
    metrics = []

    metrics_list_max = 0
    for config_dir_name in os.listdir(config_dir):
        if case_name not in config_dir_name:
            continue
        config_file = open(config_dir + config_dir_name, "r", encoding='utf-8')
        for each_line in config_file:
            metrics_list = None
            try:
                metrics_list = [i for i in each_line.strip().split(" ")]
            except:
                print(config_dir_name + ' failed')

            if 0 == metrics_list_max:
                metrics_list_max = len(metrics_list)
            elif len(metrics_list) != metrics_list_max:
                print(config_dir_name + ' len != ' + str(metrics_list_max))
                # exit(1)
            metrics += [
                {'name': config_dir_name.split('.')[0]
                    , 'version': metrics_list[0]
                    , 'CPI': float(metrics_list[1])
                    , 'bpred': float(metrics_list[2])
                    , 'Power': float(metrics_list[3])
                    , 'Area': 0.0}]
            global max_cpi, max_power
            max_cpi = max(max_cpi, float(metrics_list[1]))
            max_power = max(max_power, float(metrics_list[3]))
        config_file.close()
    return metrics


metrics_all = read_metrics('data_all_simpoint/', case_name)
# print(f"metrics_all={metrics_all}")
