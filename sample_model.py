import sys
import os
import json
import pickle
import time
import statistics
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel
import dimod.lp as lp
from compute.gantt_plot import plot_sample

import compute.bqm as bqm
import compute.cqm as cqm
import importlib
importlib.reload(bqm)
importlib.reload(cqm)

TABU_SAMPLER = "bqm-tabu"
KERBEROS_SAMPLER = "bqm-kerberos"
CLASSIC_SAMPLER = "bqm-classic"
LEAP_BQM_SAMPLER = "bqm-leap"

GENETIC_SAMPLER = "cqm-genetic"
EXACT_SAMPLER = "cqm-exact"
LEAP_CQM_SAMPLER = "cqm-leap"
SAMPLERS = [TABU_SAMPLER,
            KERBEROS_SAMPLER,
            CLASSIC_SAMPLER,
            LEAP_BQM_SAMPLER,

            GENETIC_SAMPLER,
            EXACT_SAMPLER,
            LEAP_CQM_SAMPLER]

def get_sample_energy(model_type, data, sample):
    if model_type == "bqm":
        energy = bqm.getEnergyValue(data, sample)
        return energy
    else:
        energy = cqm.getEnergyValue(data, sample)
        return energy

def get_sample_energy_cost(model_type, data, sample):
    if model_type == "bqm":
        energy_cost_sample = bqm.getEnergyCostValue(data, sample)
        return energy_cost_sample
    else:
        print("energy cost not supported for cqm")
        return 0

def get_sample_power_diff(model_type, data, sample):
    if model_type == "bqm":
        power_diff_sample = bqm.getPowerDiffValue(data, sample)
        return power_diff_sample
    else:
        print("power diff not supported for cqm")
        return 0

def analyze_sampleset(model_type, data, sampleset):
    print("analyze_sampleset...", end="")
    start_time = time.time()

    stats = {}
    unique_samples = []
    for sample in sampleset:
        if not sample in unique_samples:
            unique_samples.append(sample)
    stats["unique"] = len(unique_samples)
    best_sample = unique_samples[0]
    best_sample = {key: int(sample[key]) for key in sample}

    energies = []
    for sample in sampleset:
        energy = get_sample_energy(model_type, data, sample)
        energies.append(energy)

    minimum_energy = min(energies)
    maximum_energy = max(energies)
    median_energy = sorted(energies)[len(energies) // 2]
    mean_energy = sum(energies) / len(energies)
    stats["minimum_energy"] = minimum_energy
    stats["maximum_energy"] = maximum_energy
    stats["median_energy"] = median_energy
    stats["mean_energy"] = mean_energy
    if len(energies) > 1:
        stdev = statistics.stdev(energies)
        stats["stdev_energy"] = stdev

    if model_type == "cqm":
        return stats, best_sample

    energy_costs = []
    for sample in sampleset:
        energy_cost = get_sample_energy_cost(model_type, data, sample)
        energy_costs.append(energy_cost)
    minimum_energy_cost = min(energy_costs)
    maximum_energy_cost = max(energy_costs)
    median_energy_cost = sorted(energy_costs)[len(energy_costs) // 2]
    mean_energy_cost = sum(energy_costs) / len(energy_costs)
    stats["minimum_energy_cost"] = minimum_energy_cost
    stats["maximum_energy_cost"] = maximum_energy_cost
    stats["median_energy_cost"] = median_energy_cost
    stats["mean_energy_cost"] = mean_energy_cost
    if len(energy_costs) > 1:
        stdev = statistics.stdev(energy_costs)
        stats["stdev_energy_cost"] = stdev

    power_diffs = []
    for sample in sampleset:
        power_diff = get_sample_power_diff(model_type, data, sample)
        power_diffs.append(power_diff)
    minimum_power_diff = min(power_diffs)
    maximum_power_diff = max(power_diffs)
    median_power_diff = sorted(power_diffs)[len(power_diffs) // 2]
    mean_power_diff = sum(power_diffs) / len(power_diffs)
    stats["minimum_power_diff"] = minimum_power_diff
    stats["maximum_power_diff"] = maximum_power_diff
    stats["median_power_diff"] = median_power_diff
    stats["mean_power_diff"] = mean_power_diff

    if len(power_diffs) > 1:
        stdev = statistics.stdev(power_diffs)
        stats["stdev_power"] = stdev

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")
    return stats, best_sample

def main():
    arguments = sys.argv[1:]

    if len(arguments) < 2:
        print("Please provide model and sampler name")
        return
    if len(arguments) > 2:
        print("Too many arguments")
        return

    model_path = arguments[0]
    sampler = arguments[1]
    if sampler not in SAMPLERS:
        print("Invalid sampler name")
        print("Valid samplers: " + str(SAMPLERS))
        return

    model_type = "bqm"
    if sampler in [GENETIC_SAMPLER, EXACT_SAMPLER]:
        model_type = "cqm"

    model = None
    data = None
    with open(model_path, "rb") as f:
        input = pickle.load(f)
        model = input["model"]
        data = input["data"]

    if data["model_type"] == "cqm":
        def replace_dots(lp):
            length = len(lp)
            result = [""] * length

            in_variable = False
            for i in range(length):
                result[i] = lp[i]
                if in_variable:
                    if lp[i] == ".":
                        result[i] = "_"
                    if lp[i] == " ":
                        in_variable = False
                else:
                    if lp[i].isalpha():
                        in_variable = True
            return ''.join(result)
        model = replace_dots(model)
        model = lp.loads(model)

    print(f"selected model: {model_path}")
    print(f"selected sampler: {sampler}")
    print("sampling...")
    start = time.time()
    if sampler == TABU_SAMPLER:
        sampleset = bqm.useTabu(model, 1)
    elif sampler == KERBEROS_SAMPLER:
        sampleset = bqm.useKerberos(model, 1, os.path.basename(model_path))
    elif sampler == CLASSIC_SAMPLER:
        sampleset = bqm.useClassic(model, 1)
    elif sampler == LEAP_BQM_SAMPLER:
        sampleset = bqm.useLeap(model, 1, os.path.basename(model_path))

    elif sampler == EXACT_SAMPLER:
        sampleset = cqm.useExact(model, 1)
    elif sampler == GENETIC_SAMPLER:
        sampleset = cqm.useGenetic(model, 1)
    elif sampler == LEAP_CQM_SAMPLER:
        sampleset = cqm.useLeap(model, 1, os.path.basename(model_path))
    sampling_time = time.time() - start
    print(f"sampling time: {sampling_time}s")


    # TODO: Der Block ist nur für die Auswertung da, weil man eh nur 1 sample hat
    best_sample = sampleset[0]
    if data["model_type"] == "cqm":
        best_sample = {key.replace("_", "."): best_sample[key] for key in best_sample}
    energy = get_sample_energy(model_type, data, best_sample)
    energy_cost = 0
    power_diff = 0
    violations = 0
    if model_type == "bqm":
        energy_cost = get_sample_energy_cost(model_type, data, best_sample)
        power_diff = get_sample_power_diff(model_type, data, best_sample)
        violations = bqm.getViolations(data, best_sample)
    stats = {
        "energy": energy,
        "energy_cost": energy_cost,
        "power_diff": power_diff,
        "violations": violations
    }
    #--------------------

    cwd = os.path.dirname(os.path.realpath(__file__))
    dirs = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]
    if not "output" in dirs:
        os.mkdir(os.path.join(cwd, "output"))
    if not "plots" in dirs:
        os.mkdir(os.path.join(cwd, "plots"))

    model_index = 0
    while True:
        model_name = os.path.basename(model_path).replace(".pickle", "") + f"_{sampler}_{model_index}"
        sample_output_path = os.path.join(cwd, "output", model_name)
        plot_output_path = os.path.join(cwd, "plots", model_name)
        if os.path.exists(sample_output_path + ".json"):
            model_index += 1
        else:
            break

    sample_out = sample_output_path + ".json"
    plot_out = plot_output_path + ".png"

    output = {}
    output["model_name"] = model_name
    output["sampler"] = sampler
    output["sampling_time"] = sampling_time
    output["stats"] = stats
    output["data"] = data.copy()
    output["sample"] = best_sample
    # python keeps references to the old data-dict
    # therefore after popping the processing_durations, the reference is removed
    processing_durations = output["data"].pop("processing_durations")
    processing_durations = {str(key): processing_durations[key] for key in processing_durations}
    output["data"]["processing_durations"] = processing_durations

    with open(sample_out, "w") as file:
        file.write(json.dumps(output, indent=4))
    print(f"results written to {sample_out}")

    # TODO: remove after testing
    from math import sin, pi
    timesteps = data["timesteps"]
    x_norm = [i/timesteps for i in range(timesteps-1)] + [1.0]
    x_price = [2*pi*x for x in x_norm]
    y_price = [0.5*sin(x)+1 for x in x_price]
    data["price"] = y_price

    plot_sample(best_sample, data, filename=plot_out, color_only=True)

if __name__ == "__main__":
    main()