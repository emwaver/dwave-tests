import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import json

BQM_MODEL = "bqm"
CQM_MODEL = "cqm"

IDLE = "\u231B" #⌛
MAINTENANCE = "\U0001F527" #🔧
SETUP = "\U0001F6E0" #🛠️
PROCESS = ""
LIGHT = "\U0001F506" #🔆

# Unicode Symbols are not working on Elwetritsch
IDLE = "i" #⌛
MAINTENANCE = "m" #🔧
SETUP = "s" #🛠️
PROCESS = ""
LIGHT = "1" #🔆

gantt_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan"
]

def plot_sample(sample, data, color_only=False, filename="plot.png"):
    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    model_type = data["model_type"]

    if len(gantt_colors) < len(jobs):
        print("Not enough base colors for all jobs. Consider adding a few reasonable colors to the color list.")
        print("Using random colors instead.")
        while len(gantt_colors) < len(jobs):
            gantt_colors.append('#%06X' % random.randint(0, 0xFFFFFF))

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]})
    ticks_interval = int(timesteps / 30) + 1
    ax[0].set_xlim([0, timesteps])
    ax[0].set_xticks(np.arange(0, timesteps, ticks_interval))

    ax[1].set_xlim([0, timesteps])
    ax[1].set_xticks(np.arange(0, timesteps, ticks_interval))
    ax[1].set_yticks(np.arange(0, len(machines)+1, 1))
    ax[1].set_yticklabels([f"M{m}" for m in range(len(machines))] + ["Facility"])
    ax[1].set_ylim([-1, len(machines)+1])
    ax[1].invert_yaxis()
    ax[1].set_axisbelow(True)
    ax[1].grid(axis='y')

    machine_states = None
    facility_states = None
    if model_type == BQM_MODEL:
        sample, machine_states, facility_states = read_bqm_sample(sample, data)
    else:
        sample, machine_states, facility_states = read_cqm_sample(sample, data)

    # count concurrent operations
    concurrent = {}
    for key in sample:
        values = key.split(".")
        if values[0] != 'x':
            continue
        i, j, m, t, v = [int(value) for value in values[1:]]

        concurrent[(m, t)] = concurrent.get((m, t), 0) + sample[key]

    # draw operation bars
    displayed = {}
    handles = {}
    for key in sample:
        if sample[key] == 0:
            continue

        values = key.split(".")
        if values[0] != 'x':
            continue
        i, j, m, t, v = [int(value) for value in values[1:]]

        x_range = (t, 1)
        already_displayed = displayed.get((m, t), 0)
        y_height = 0.8 / concurrent[(m, t)]
        y_start = m -0.4 + y_height * already_displayed
        displayed[(m, t)] = already_displayed + 1

        handles[i] = ax[1].broken_barh(xranges=[x_range],
                                       yrange=(y_start, y_height),
                                       facecolors=gantt_colors[i])

    # draw machine states
    font = {"fontname": "Segoe UI Emoji"}
    for m in machines:
        for t in range(timesteps):
            text = machine_states[m][t]
            if color_only:
                if text == IDLE:
                    ax[1].broken_barh(xranges=[(t, 1)],
                                        yrange=(m - 0.4, 0.8),
                                        facecolors="lightgray")
                if text == MAINTENANCE:
                    ax[1].broken_barh(xranges=[(t, 1)],
                                      yrange=(m - 0.4, 0.8),
                                      facecolors="lightgray",
                                      hatch='xxx')
                if text == SETUP:
                    ax[1].broken_barh(xranges=[(t, 1)],
                                    yrange=(m - 0.4, 0.8),
                                    facecolors="gray")
            else:
                ax[1].text(t+0.5, m, text, fontsize=10, horizontalalignment='center', verticalalignment='center', **font)

    # draw facility states
    for t in range(timesteps):
        text = facility_states[t]
        if color_only:
            ax[1].broken_barh(xranges=[(t, 1)],
                              yrange=(len(machines) - 0.4, 0.8),
                              facecolors="lightgray")
        else:
            ax[1].text(t+0.5, len(machines), text, fontsize=10, horizontalalignment='center', verticalalignment='center', **font)

    # draw power chart
    x_values = [t for t in range(timesteps + 1)]
    power = [0] * timesteps
    for m in machines:
        p_idle = machines[m]["idle_power"]
        p_setup = machines[m]["setup_power"]
        p_maint = machines[m]["maintenance_power"]

        for t in range(timesteps):
            if machine_states[m][t] == IDLE:
                power[t] += p_idle
            if machine_states[m][t] == SETUP:
                power[t] += p_setup
            if machine_states[m][t] == MAINTENANCE:
                power[t] += p_maint
            pass

    for key in sample:
        values = key.split(".")
        if values[0] != 'x':
            continue
        i, j, m, t, v = [int(value) for value in values[1:]]

        p_proc = machines[m]["speeds"][v]["processing_power"]
        efficiency = machines[m]["speeds"][v]["efficiency"]
        energy = jobs[i][j]["energy"]
        duration = processing_durations[(i,j,m,v)]

        power[t] += p_proc
        power[t] += energy / duration / efficiency

    power_diff = 0
    for t in range(timesteps):
        if not (t > 0):
            continue
        t_last = t - 1
        power_diff += (power[t] - power[t_last])**2
    # print("measured power diff:", power_diff)

    ax[0].set_ylabel("kW", color="black")
    ax[0].stairs(edges=x_values, values=power, color="black", linewidth=1)

    # draw price chart
    # from math import pi
    # y_price = data["price"]
    # ax_twin = ax[0].twinx()
    # ax_twin.set_ylabel("€/kWh", color="red")
    # print(y_price)
    # ax_twin.stairs(edges=x_values, values=y_price, color="red", linewidth=1)

    # draw legend
    try:
        highest_handle = max(handles.keys())
    except:
        highest_handle = 1
    highest_handle_digits = len(str(highest_handle))

    handles_list = list(handles.items())
    handles_list.sort(key=lambda handle: f"Job {'0' * (highest_handle_digits - len(str(handle[0])))}{handle[0]}")
    o_handles = []
    o_labels = []
    for label, handle in handles_list:
        o_handles.append(handle)
        o_labels.append(f"Job {'0' * (highest_handle_digits - len(str(label)))}{label}")
    ax[1].legend(handles=o_handles, labels=o_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    if filename != "":
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def read_bqm_sample(sample, data):
    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]

    sample = {key: sample[key] for key in sample if sample[key] == 1}
    machine_states = {}
    for m in machines:
        machine_states[m] = [IDLE] * timesteps
    facility_states = [LIGHT] * timesteps

    warning = False
    # expand sampleset from starting points
    sample_copy = sample.copy()
    for key in sample:
        values = key.split(".")
        if values[0] != 'x':
            continue
        i, j, m, t, v = [int(value) for value in values[1:]]
        duration = processing_durations[(i,j,m,v)]
        for t2 in range(t, t + duration):
            if t2 >= timesteps:
                warning = True
                continue

            sample_copy[f"x.{i}.{j}.{m}.{t2}.{v}"] = 1
            machine_states[m][t2] = f"{j}\nv{v}"
        for t2 in range(t - machines[m]["setup_duration"], t):
            machine_states[m][t2] = SETUP
    sample = sample_copy

    # read maintenance states
    for key in sample:
        values = key.split(".")
        if values[0] != 'maintenance':
            continue
        m, t = [int(value) for value in values[1:]]
        maintenance_duration = machines[m]["maintenance_duration"]
        for t2 in range(maintenance_duration):
            if t + t2 >= timesteps:
                warning = True
                continue
            machine_states[m][t+t2] = MAINTENANCE
    if warning:
        print("WARNING: sample contains values outside of timesteps")

    # read turnon states:
    for key in sample:
        values = key.split(".")
        if values[0] != 'turnon':
            continue
        m, t = [int(value) for value in values[1:]]

        for t2 in range(t):
            machine_states[m][t2] = ""

    # read turnoff states:
    for key in sample:
        values = key.split(".")
        if values[0] != 'turnoff':
            continue
        m, t = [int(value) for value in values[1:]]

        for t2 in range(t, timesteps):
            machine_states[m][t2] = ""

    # read facility states
    for key in sample:
        values = key.split(".")
        if values[0] != 'facility_off':
            continue
        t = int(values[1])

        for t2 in range(t, timesteps):
            facility_states[t2] = ""

    return sample, machine_states, facility_states

def read_cqm_sample(sample, data):

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]

    machine_states = {}
    for m in machines:
        machine_states[m] = [IDLE] * timesteps
    facility_states = [LIGHT] * timesteps

    binary_sample = {}
    for key in sample:
        values = key.split(".")
        if values[0] == "maintenance":
            binary_sample[key] = sample[key]
        if values[0] == "turnoff":
            binary_sample[key] = sample[key]
        if values[0] == "turnon":
            binary_sample[key] = sample[key]
        if values[0] == "facility":
            binary_sample[key] = sample[key]

    # convert cqm solution to bqm solution
    operation_keys = []
    for key in sample:
        values = key.split(".")
        if values[0] != 'start':
            continue
        i, j = [int(value) for value in values[1:]]
        operation_keys.append((i, j))

    warning = False
    for operation in operation_keys:
        i, j = operation

        start_ij = sample[f"start.{i}.{j}"]
        found = False
        duration = 0
        for m in machines:
            for v in machines[m]["speeds"]:
                if sample[f"x.{i}.{j}.{m}.{v}"] == 1:
                    found = True
                    duration = processing_durations[(i,j,m,v)]
                    break
            if found:
                break
        if not found:
            continue

        setup_duration = machines[m]["setup_duration"]
        for d in range(setup_duration):
            if start_ij - d - 1 < 0:
                warning = True
                break
            machine_states[m][start_ij-d-1] = SETUP
        for d in range(duration):
            binary_sample[f"x.{i}.{j}.{m}.{start_ij+d}.{v}"] = 1
            if start_ij + d >= timesteps:
                warning = True
                break
            machine_states[m][start_ij+d] = f"{j}\nv{v}"
    if warning:
        print("WARNING: sample contains values outside of timesteps")

    sample = binary_sample.copy()

    # read maintenance states
    for key in sample:
        values = key.split(".")
        if values[0] != 'maintenance':
            continue
        m = int(values[1])

        maintenance_start = sample[key]
        maintenance_duration = machines[m]["maintenance_duration"]
        for t in range(maintenance_duration):
            machine_states[m][maintenance_start + t] = MAINTENANCE

    # read turnoff states
    for key in sample:
        values = key.split(".")
        if values[0] != 'turnoff':
            continue
        m = int(values[1])

        turnoff_start = sample[key]
        for t in range(turnoff_start, timesteps):
            machine_states[m][t] = ""

    # read turnon states
    for key in sample:
        values = key.split(".")
        if values[0] != 'turnon':
            continue
        m = int(values[1])

        turnon_start = sample[key]
        for t in range(turnon_start):
            machine_states[m][t] = ""

    # read facility states
    facility_turnoff = sample["facility"]
    for t in range(facility_turnoff, timesteps):
        facility_states[t] = ""

    return sample, machine_states, facility_states