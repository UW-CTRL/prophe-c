from decision_making.util_classes import AgentState, PlannerParams
from decision_making.util_functions import max_likelihood, kl_divergence, obs_env_prep, combined
from decision_making.gen_models import query_priors_gen, query_from_house_data, uniform_prior
from decision_making.bayes_models import ObjBayesNet
import decision_making.planner as p
from house_env.sim import Sim
from copy import deepcopy
import pandas as pd
import pickle
import networkx as nx
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re


MARKERS = {"Max Likelihood": ('o', 'blue'), "Random": ('o', 'orange'), 
            "Ours": ('x', 'red'), "From Data": ('o', 'green')}

MARKERS_TIMEDICT = {"Max Likelihood (small model)": ('x', 'blue', (0, (5, 10))), 
                    "Max Likelihood (medium model)": ('o', 'blue', (0, (1, 10))),
                    "Max Likelihood (large model)": ('*', 'blue', '-'), 
                    "Ours (small model)": ('x', 'red', (0, (5, 10))),
                    "Ours (medium model)": ('o', 'red', (0, (1, 10))),
                    "Ours (large model)": ('*', 'red', '-'),
                    "Random": ('*', 'orange', '-'), 
                    "Domain Specific": ('*', 'green', '-')
                    }

# line_widths = {'Data1': 2.0, '

# TODO merge this file with logic in sim.modify_graph()
def load_world_for_testing(file_path:str):
    # Load the map + scene-graph
    storage_dict = pickle.load(open(file_path, 'rb'))
    meta_data = storage_dict["data"]
    scene_graph =  nx.Graph(storage_dict["scene_graph"])
    
    # Add node-attributes to all the nodes in the scene-graph so
    # we can access them through the planner easily
    attr_dict = {}
    for node in meta_data:
        connection_surfaces = []
        if meta_data[node]["class"] == "room":
            connection_surfaces = meta_data[node]["doorways"]

        attr_dict[meta_data[node]["name"]] = {"class": meta_data[node]["class"], "centroid": meta_data[node]["centroid"], 
                                            "contour": meta_data[node]["contour"], "connection_surfaces": connection_surfaces}
    nx.set_node_attributes(scene_graph, attr_dict)

    return scene_graph


def run_obj_search_tests(tests_per_env, num_envs, file_path, pid):
    """
    Conducts ablations on each parameter to see how performance is impacted by sweeping the values
    of each parameter individually, while maintaining medium "default" values for other parameters.
    :param tests_per_env: How many times we should evaluate parameter effects for a single environment/level
    :param env_sample_indices: A list of file IDs to load scene-graph environments from.
    :param file_path: Location of scene-graph files to import
    :param pid: the ID of the pool that is running this function.
    """
    # cost_scaler: [divergence, likelihood, distance]
    # ML_TEST = {"cost_scaler": [0, 1, 0], "horizon_len": 8, "traj_samples": 1}
    # LAZY_TEST = {"cost_scaler": [0, 0, 1], "horizon_len": 5, "traj_samples": 500}
    # MYOPIC = {"cost_scaler": [0.001, 100, 1], "horizon_len": 1, "traj_samples": 500}
    MINE = {"cost_scaler": [0, 1, 1], "horizon_len": 3, "traj_samples": 50}
    # RANDOM = {"cost_scaler": [0, 0, 0], "horizon_len": 0, "traj_samples": 0}
    to_test = MINE

    rd.seed(pid)
    root = "house"
    random_indices = [rd.randint(0, 999) for _ in range(num_envs)]
    
    results = {"success":(), "cost":(), "time":()}
    
    # TODO: it would be nice to eliminate the need for this lambda function
    check_obj_validity = lambda node_label, G: any("furniture" in G.nodes[neighbor]['class'] \
                                                    for neighbor in G.neighbors(node_label))

    for idx, env_id in enumerate(random_indices):
        graph = load_world_for_testing(f'/home/isaac/isogrids/src/house_env/houses/house{env_id}.pickle')
        
        print(f"LOADED NEW ENVIRONMENT FOR PROCESS {pid}...")

        for i in range(tests_per_env):
            start_point, starting_graph = obs_env_prep(graph, i, po=False)
            init_state = AgentState(start_point, graph.nodes[start_point]["centroid"])

            objs = [node for node in graph.nodes if nx.get_node_attributes(graph, "class")[node] == "object"]
            valid_objs = [obj for obj in objs if check_obj_validity(obj, graph)]
            rd.seed(i)
            obj = rd.choice(valid_objs)[:-1] # remove end digit identifier
            print(f"FINDING OBJ: {obj}")
            rd.seed()
            obj = " ".join(re.findall("[a-zA-Z]+", obj)) # make sure that object-string doesn't have any dashes and such

            bn = ObjBayesNet

            params = PlannerParams(deepcopy(init_state), deepcopy(starting_graph), obj, root, to_test["cost_scaler"], to_test["horizon_len"],
                                 to_test["traj_samples"], combined, {0, 1})

            sim_obs = Sim(graph)
            plan = p.Planner(params, query_priors_gen, bn, lambda obs:True if obs == 1 else False,
                            external_controller=sim_obs.decision_outcome, true_graph=graph, greedy_policy=False)
            success, cost, time = plan.planner()

            curr_succ = results["success"]
            curr_cost = results["cost"]
            curr_time = results["time"]
            results["success"] = curr_succ + (success,)
            results["cost"] = curr_cost + (cost,)
            results["time"] = curr_time + (time,)

            print("---------------------------------------------------------------------------------")
            print(f"FINISHED {(idx * tests_per_env) + (i + 1)} TESTS OUT OF {num_envs * tests_per_env} TOTAL TESTS")
            # TODO: compute lower cost bound for this environment and this specific search-task

            # TODO: save results for each reward regime
    pickle.dump(results, open(file_path + str(pid) + '.pickle', 'wb'))


def hists_plotter(files:dict, type:str, po:bool):
    size_metrics_dict ={}
    for key in files.keys():
        size_metrics_dict[key] = get_data(files[key])
    if po:
        plot_all_time_hist(size_metrics_dict, "Map Partially Observed " + type, type, po)
    else:
        plot_all_time_hist(size_metrics_dict, "Map Fully Known " + type, type, po)
    
    # for key in size_metrics_dict.keys():


def time_dist_plotter(files, po: bool):
    metrics_dict = get_data(files)
    plot_time_distance({method:metrics_dict[method]["Leaf Steps"] for method in metrics_dict.keys()}, 
                       {method:metrics_dict[method]["Distance"] for method in metrics_dict.keys()},
                       po)


def get_data(files:dict):
    metrics_dict = {}
    PARAM = "cost_scaler"
    for file in files.keys():
        success_total = ()
        cost_total = ()
        time_total = ()
        for i in range(24):

            results = pickle.load(open(files[file] + f"normal_test{i}.pickle", 'rb'))
            success = results["success"]
        
            cost = results["cost"]
            
            time = results["time"]
            
            success_total += success
            cost_total += cost
            time_total += time

        metrics_dict[file] = {"success" : success_total, "Distance": cost_total, "Leaf Steps": time_total}

    return metrics_dict


def plot_time_distance(time_dict, distance_dict, po):
    title = "Partially Observed" if po else "Fully Known"
    time_data = [val for val in time_dict.values()]
    dist_data = [val for val in distance_dict.values()]
    labels = [key for key in time_dict.keys()]
    norm = 10000
    if not po:
        norm = 14000
    plt.figure(figsize=(10, 6))    
    for idx, (time, dist, label) in enumerate(zip(time_data, dist_data, labels)):
        dist = np.array(dist) / (norm)
        coeff = np.polyfit(time, dist, 1)  # Fit a first-degree (linear) polynomial
        line = np.poly1d(coeff)
        linewidth = 2
        if label == "Domain Specific":
            linewidth=4
        plt.plot(time, line(time), linewidth=linewidth, label=label, color=MARKERS_TIMEDICT[label][1], linestyle=MARKERS_TIMEDICT[label][2],
                    zorder=idx + 1)

    plt.xlabel('Leaf Steps')
    plt.ylabel('Distance')
    plt.title(f'Leaf Steps vs. Distance, Map {title}')
    plt.subplots_adjust(top=0.9)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig(f"Distance vs Time, {title}.png")
    plt.clf()


def plot_all_time_hist(all_data, title, metric, po):
    # Create subplots for the histograms
    plt.rc('font', size=16)
    fig, axs = plt.subplots(len(all_data), 1, figsize=(12, 10))
    # plt.grid()
    bins = []
    y_scale = ()

    if metric == "Leaf Steps":
        bins = np.arange(1, 15, 1)
        if po:
            y_scale = (0, 50)
        else:
            y_scale = (0, 120)
    else:
        bins = np.arange(100, 8000, 1000)
        bins = np.arange(0, 1, 0.1)
        if po:
            y_scale = (0, 140)
            # plt.rc('font', size=16)
        else:
            y_scale = (0, 170)

    # Calculate the positions for the bars in each category
    for idx, size in enumerate(all_data.keys()):
        data = tuple([all_data[size][method][metric] for method in all_data[size].keys()])
        if metric == "Distance":
            data = tuple([np.array(d) / 10000 for d in data])
        labels = tuple([method for method in all_data[size].keys()])
        axs[idx].hist(data, bins=bins, 
                        edgecolor='black', alpha=0.5, color=[MARKERS[l][1] for l in labels], label=labels)
        
        axs[idx].set_title(f'{size}')
        axs[idx].set_xlabel(f'{metric}')
        axs[idx].set_ylabel('Frequency')
        axs[idx].legend()
        axs[idx].grid()
        # axs[idx].set
        axs[idx].set_ylim(*y_scale)

        for d, l, in zip(data, labels):
            # if metric == "Distance":
            #     d = np.array(d)/ 1000
            axs[idx].axvline(np.mean(d), color=MARKERS[l][1], linestyle='dashed')

    # Adjust layout and display the plots
    plt.tight_layout()
    context = ""
    context = "Map Partially Observed" if po else "Map Fully Known"
    plt.suptitle(f'{metric} vs Frequency, {context}', fontsize=20, x=0.54)
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"{title}.png")
    plt.clf()






