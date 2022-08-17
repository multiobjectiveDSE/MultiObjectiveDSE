import copy
from copy import deepcopy

import numpy as np

from simulation_metrics import get_var_list_index
from config import case_name

if 0:
    from ax import *


    def get_real_pareto_frontier(experiment, metric_a, metric_b, obj_list):

        model = Models.SOBOL(
            search_space=experiment.search_space,
            seed=1234,
        )
        # from ax.modelbridge.factory import get_factorial
        # m = get_factorial(search_space)
        # gr = m.gen(n=10)
        if 0:
            trials = [
                {"DispatchWidth": 0, "ExeInt": 0, "ExeFP": 0, "LSQ": 0, "Dcache": 0, "Icache": 0, "BP": 0,
                 "L2cache": 0},
            ]
            for i, trial in enumerate(trials):
                arm_name = f"{i}_0"
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters=trial))
                if 0:
                    data = Data(df=pd.DataFrame.from_records([
                        {
                            "arm_name": arm_name,
                            "metric_name": metric_name,
                            "mean": output["mean"],
                            "sem": output["sem"],
                            "trial_index": i,
                        }
                        for metric_name, output in trial["output"].items()
                    ])
                    )
                    experiment.attach_data(start_data)
                trial.run().complete()

            # experiment.attach_trial(start_data)
        experiment.new_batch_trial(model.gen(30)).run()

        # for parameter in experiment.search_space.parameters:
        # print(f"parameter:{parameter}")

        if 1:
            from ax.modelbridge.factory import get_MOO_EHVI
            # Fit a GP-based model & EHVI
            model_dumpy = get_MOO_EHVI(
                experiment=experiment,
                data=experiment.fetch_data(),
            )

        # ## Plot Pareto Frontier based on model posterior
        #
        # The plotted points are samples from the fitted model's posterior, not observed samples.

        if 1:
            # print("compute_posterior_pareto_frontier")
            from ax.plot.pareto_utils import compute_posterior_pareto_frontier
            frontier = compute_posterior_pareto_frontier(
                experiment=experiment,
                data=experiment.fetch_data(),
                primary_objective=metric_b,
                secondary_objective=metric_a,
                absolute_metrics=obj_list,
                num_points=20,
            )

            print("plot_pareto_frontier......")
            from ax.utils.notebook.plotting import render
            from ax.plot.pareto_frontier import plot_pareto_frontier
            render(plot_pareto_frontier(frontier, CI_level=0.90))

        try:
            from ax.modelbridge.modelbridge_utils import observed_hypervolume
            hv = observed_hypervolume(modelbridge=model_dumpy)
        except:
            hv = 0
            print("Failed to compute hv")
        print(f"real_pareto_frontier hv={hv}")
        return hv


def is_pareto_efficient_dumb(sample_origin, costs_origin):
    'returns Pareto efficient row subset of pts'
    # sort points by decreasing sum of coordinates
    # print(f"costs.sum(1)={costs.sum(1).argsort()}")
    # print(f"sample_origin={sample_origin}, costs_origin={costs_origin}")
    index = costs_origin.sum(1).argsort()
    # print(f"index={index}")
    costs = deepcopy(costs_origin)[index]
    # print(f"origin costs={costs}")
    # print(f"sample_origin={sample_origin}")
    sample = np.array(deepcopy(sample_origin))[index]
    # print(f"sample_origin={sample}, costs_origin={costs}")
    # print(f"costs={costs}")
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i in range(costs.shape[0]):
        # print(f"costs.shape={costs.shape}")
        # process each point in turn
        n = costs.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        # print(f"costs[i+1:]={costs[i+1:]}")
        # print(f"costs[i]={costs[i]}")
        # print(f"(costs[i+1:] < costs[i])={(costs[i+1:] < costs[i])}")
        is_efficient[i + 1:n] = (costs[i + 1:] < costs[i]).any(1)
        is_efficient[i] = True
        # keep points undominated so far
        # print(f"i={i}, n={n}, is_efficient={is_efficient[:n]}")
        costs = costs[is_efficient[:n]]
        # print(f"costs={costs}")
        sample = sample[is_efficient[:n]]
        index = index[is_efficient[:n]]
    if False:
        import matplotlib.pyplot as plt
        plt.scatter(costs_origin[:, 0:1], costs_origin[:, 1:2], c="gray", s=10)
        plt.scatter(costs[:, 0:1], costs[:, 1:2], c="red", marker="x", s=16)
        plt.show()
        exit(1)
    return sample, costs


def scale_for_hv(obj_values, ref_point):
    obj_values_scale = copy.deepcopy(obj_values)
    obj_values_scale[:, 0] = obj_values[:, 0] / ref_point[0]
    obj_values_scale[:, 1] = obj_values[:, 1] / ref_point[1]
    return obj_values_scale

def get_pareto_optimality_from_file(filename):
    file = open(filename, 'r')
    pareto_points_x = []
    pareto_points_y = []
    points_config = []
    sample_num_i = 0
    for point in file:
        point_str = point.split(' ')
        if 1 < len(point_str):
            cpi = float(point_str[0])
            pareto_points_x.append(cpi)
            power = float(point_str[1])
            pareto_points_y.append(power)
            points_config.append(point_str[2].strip('\n'))
        elif 1 == len(point_str):
            sample_num_i = int(point.strip('\n'))
    file.close()
    return [pareto_points_x, pareto_points_y, points_config, sample_num_i]


def generate_pareto_optimality_perfect_filename(case_name, area_mode):
    pareto_optimality_perfect_filename = 'real_pareto_optimality/'  # "../"
    # global case_name
    pareto_optimality_perfect_filename += case_name + "_"
    if area_mode:
        pareto_optimality_perfect_filename += 'area-pareto_optimality_perfect.txt'
    else:
        pareto_optimality_perfect_filename += 'power-pareto_optimality_perfect.txt'
    return pareto_optimality_perfect_filename


def get_pareto_optimality_from_file_ax_interface(case_name):
    pareto_optimality_perfect_filename = generate_pareto_optimality_perfect_filename(case_name, area_mode=0)
    [real_pareto_points_x, real_pareto_points_y, real_pareto_points_config,
     sample_num_i] = get_pareto_optimality_from_file(pareto_optimality_perfect_filename)
    config_vector_list = []
    for config in real_pareto_points_config:
        # print(config)
        config_vector = transfer_version_to_var(config)
        config_vector_list.append(config_vector)
    # print(config_vector_list)
    return [real_pareto_points_x, real_pareto_points_y, config_vector_list]


def get_dataset(case_name):
    raw_data_file = open("data_all_simpoint/" + case_name + '.txt', 'r')
    version_list = []
    y = []
    for each_line in raw_data_file:
        each_line_str = each_line.split(' ')
        version = each_line_str[0]
        if 30 < len(version):
            cpi = float(each_line_str[1])
            power = float(each_line_str[3])
            version_list.append(version)
            y.append([cpi, power])
    raw_data_file.close()
    return version_list, y

def cal_pareto_optimality_to_file(case_name):
    pareto_optimality_perfect_filename = generate_pareto_optimality_perfect_filename(case_name, area_mode=0)

    version_list, y = get_dataset(case_name=case_name)

    real_pareto_points_x, real_pareto_points_y = is_pareto_efficient_dumb(np.asarray(version_list), np.asarray(y))

    file = open(pareto_optimality_perfect_filename, 'w')
    file.write(f"{1}\n")  # header
    for each_x, each_y in zip(real_pareto_points_x, real_pareto_points_y):
        file.write(f"{each_y[0]} {each_y[1]} {each_x}\n")
    file.close()


def transfer_version_to_var(config):
    config_vector = [int(config[var_index]) for var_index in get_var_list_index()]
    return config_vector


def plot_histogram(case_name):
    version_list, y = get_dataset(case_name=case_name)
    data = np.asarray(y)[:,0]
    #print(np.shape(data))
    #density=False (default), return the num; True, return probility density
    # len(bin_edges) == len(hist) + 1
    #hist, bin_edges = np.histogram(a=y[:,0], bins=20)

    import matplotlib.pyplot as plt
    # matplotlib.axes.Axes.hist() 方法的接口
    freq, bins, patches = plt.hist(x=data, range=(0,np.max(data)), bins='auto', color='darkblue', alpha=0.7, rwidth=0.85)
    print(freq)
    #print(bins)
    freq_sum = np.sum(freq)
    print(f"freq_sum={freq_sum}, bins = {len(freq)}")
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('CPI histogram of ' + case_name)
    maxfreq = freq.max()
    plt.xticks(bins, fontsize=5, rotation=90)
    #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    fig_name = "fig/histogram/" + "CPI_histogram_" + case_name
    fig_name += ".png"
    plt.savefig(fig_name)

if __name__ == '__main__':
    #print(f"cal_pareto_optimality_to_file {case_name}")
    #cal_pareto_optimality_to_file(case_name=case_name)
    plot_histogram(case_name=case_name)
