import sys
import csv
import os
import random
import math
import time
import json
import pickle
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel
import dimod.lp as lp

import compute.bqm as bqm
import compute.cqm as cqm
import importlib
importlib.reload(bqm)
importlib.reload(cqm)

USAGE = f"""
Usage: python create_model.py
    -o                  set number of total operations
    -b                  set maximum number of operations per job
    -m                  set number of machines
    -v                  set number of additional speeds (randomly assigned to machines)
    -sm                 set maximum setup duration
    -mm                 set maximum maintenance duration
    -t                  set total number of timesteps
    -tb                 set buffer time, tb=10 reserves 10 free timesteps (subtracting from t)
    -f                  set facility power

    -e                  add total energy objective with reciprocal weight
    -c                  add energy cost objective with reciprocal weight
    -p                  add power smoothing objective with reciprocal weight

    -bqm                create binary quadratic model
    -cqm                create constrained quadratic model
"""

def createRandomData(total_operations,
                     job_operations,
                     total_machines,
                     additional_speeds,
                     max_setup_duration,
                     max_maintenance_duration,
                     timesteps):
    print("createRandomData... ", end="")
    start_time = time.time()
    random.seed(1)

    setup_durations = [0] * total_machines
    maintenance_durations = [0] * total_machines
    available_times = [timesteps] * total_machines
    cuts = [0] * total_machines

    for i in range(total_machines):
        setup_durations[i] = random.randint(1, max_setup_duration)
        maintenance_durations[i] = random.randint(1, max_maintenance_duration)
        available_times[i] -= setup_durations[i] + maintenance_durations[i]

    for cut in range(total_operations - total_machines):
        cut_machine_order = random.sample(range(total_machines), total_machines)

        cut_placed = False
        for i in range(total_machines):
            cut_machine = cut_machine_order[i]
            setup_duration = setup_durations[cut_machine]

            if available_times[cut_machine] > setup_duration:
                cuts[cut_machine] += 1
                available_times[cut_machine] -= setup_duration + 1
                cut_placed = True
                break
        if not cut_placed:
            print("No more cuts available")
            print("change parameters and try again")
            return None, None

    MAINTENANCE = "maintenance"
    OPERATION = "operation"

    partitions = []
    for m in range(total_machines):
        setup_duration = setup_durations[m]
        min_time = setup_duration + 1

        parts = [{
            "type": OPERATION,
            "duration": min_time,
            "machine": m,
        } for i in range(cuts[m] + 1)]

        available_time = timesteps
        available_time -= maintenance_durations[m]
        available_time -= min_time * (cuts[m] + 1)

        for t in range(available_time):
            index = random.randint(0, cuts[m])
            parts[index]["duration"] += 1

        partitions.append(parts)

    for m, parts in enumerate(partitions):
        maintenance_index = random.randint(0, len(parts))
        maintenance = {
            "type": MAINTENANCE,
            "duration": maintenance_durations[m]
        }
        parts.insert(maintenance_index, maintenance)

    for parts in partitions:
        t = 0
        for part in parts:
            part["start"] = t
            t += part["duration"]
            part["end"] = t - 1

    operations = []
    for parts in partitions:
        for part in parts:
            operations.append(part)
    operations = [operation for operation in operations if operation["type"] != MAINTENANCE]

    jobs = {}
    while len(operations) > 0:
        index = random.randint(0, len(operations) - 1)
        op_select = operations.pop(index)

        assigned = False
        for job_index in jobs:
            job = jobs[job_index]
            if len(job) >= job_operations:
                continue
            overlap = False
            for op_index in job:
                operation = job[op_index]
                start1 = operation["start"]
                end1 = operation["end"]
                start2 = op_select["start"]
                end2 = op_select["end"]

                if (start2 - end1) * (start1 - end2) >= 0:
                    overlap = True
                    break
            if not overlap:
                job[len(job)] = op_select
                assigned = True
                break
        if not assigned:
            jobs[len(jobs)] = {0: op_select}

    for job_index in jobs:
        job = jobs[job_index]
        operations = job.values()
        operations = sorted(operations, key=lambda x: x["start"])
        op_index = 0
        for operation in operations:
            jobs[job_index][op_index] = operation
            op_index += 1

    # print()
    # print(json.dumps(jobs, indent=4))

    machine_speeds = [1]*total_machines
    for i in range(additional_speeds):
        machine_speeds[random.randint(0, total_machines-1)] += 1

    machines = {}
    for m in range(total_machines):
        machines[m] = {
            "idle_power": random.randint(1, 3),
            "maintenance_power": random.randint(1, 3),
            "maintenance_duration": maintenance_durations[m],
            "setup_power": random.randint(1, 3),
            "setup_duration": setup_durations[m],
            "speeds": {}
        }
        for s in range(machine_speeds[m]):
            machines[m]["speeds"][s] = {
                "speed": random.randint(5, 10),
                "processing_power": random.randint(3, 5),
                "efficiency": random.randint(50, 100) / 100
            }

    for i in jobs:
        for j in jobs[i]:
            total_duration = jobs[i][j]["duration"]
            m = jobs[i][j]["machine"]
            processing_duration = total_duration - machines[m]["setup_duration"]
            speed = machines[m]["speeds"][0]["speed"]

            steps = processing_duration * speed
            jobs[i][j]["steps"] = steps
            jobs[i][j]["energy"] = random.randint(5, 15)

            jobs[i][j].pop("type")
            jobs[i][j].pop("duration")
            jobs[i][j].pop("machine")
            jobs[i][j].pop("start")
            jobs[i][j].pop("end")

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")
    return jobs, machines

def parse_files(job_file, machine_file):
    jobs = {}
    machines = {}

    with open(job_file) as file:
        reader = csv.reader(file, delimiter=';')
        for i, row in enumerate(reader):
            if i == 0:
                continue

            job, operation, steps, energy = row
            job = int(job)
            operation = int(operation)
            steps = float(steps)
            energy = float(energy)

            jobs[job] = jobs.get(job, {})
            jobs[job][operation] = jobs[job].get(operation, {})

            jobs[job][operation]["steps"] = steps
            jobs[job][operation]["energy"] = energy

    with open(machine_file) as file:
        reader = csv.reader(file, delimiter=';')
        for i, row in enumerate(reader):
            if i == 0:
                continue

            (
                machine_id,
                idle_power,
                maintenance_power,
                maintenance_duration,
                setup_power,
                setup_duration,
                speed_id,
                speed,
                processing_power,
                efficiency
            ) = row
            machine_id = int(machine_id)
            idle_power = float(idle_power)
            maintenance_power = float(maintenance_power)
            maintenance_duration = int(maintenance_duration)
            setup_power = float(setup_power)
            setup_duration = int(setup_duration)
            speed_id = int(speed_id)
            speed = float(speed)
            processing_power = float(processing_power)
            efficiency = float(efficiency)

            machines[machine_id] = machines.get(machine_id, {})
            machines[machine_id]["idle_power"] = idle_power
            machines[machine_id]["maintenance_power"] = maintenance_power
            machines[machine_id]["maintenance_duration"] = maintenance_duration
            machines[machine_id]["setup_power"] = setup_power
            machines[machine_id]["setup_duration"] = setup_duration
            machines[machine_id]["speeds"] = machines[machine_id].get("speeds", {})
            machines[machine_id]["speeds"][speed_id] = machines[machine_id]["speeds"].get(speed_id, {})
            machines[machine_id]["speeds"][speed_id]["speed"] = speed
            machines[machine_id]["speeds"][speed_id]["processing_power"] = processing_power
            machines[machine_id]["speeds"][speed_id]["efficiency"] = efficiency

    return jobs, machines

def calculate_op_durations(jobs, machines):
    result = {}

    for i in jobs:
        for j in jobs[i]:
            steps = jobs[i][j]["steps"]

            for m in machines:
                speeds = machines[m]["speeds"]
                for v in speeds:
                    speed = speeds[v]["speed"]
                    duration = math.ceil(steps / speed)
                    result[(i,j,m,v)] = duration

    return result

def get_valid_intervals(data):

    durations = data["processing_durations"]
    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]

    # initialize data structure
    valid_intervals = {}
    for i in jobs:
        valid_intervals[i] = {}
        for j in jobs[i]:
            valid_intervals[i][j] = {}
            valid_intervals[i][j]["start"] = 0
            valid_intervals[i][j]["end"] = timesteps

    # calculate minimum processing and execution times
    min_setup_duration = min([machines[m]["setup_duration"] for m in machines])
    min_times = {}
    for i in jobs:
        for j in jobs[i]:
            min_times[(i,j)] = {}

            min_processing_time = float("inf")
            min_execution_time = float("inf")
            for m in machines:
                speeds = machines[m]["speeds"]
                for v in speeds:
                    processing_time = durations[(i,j,m,v)]
                    execution_time = machines[m]["setup_duration"] + processing_time
                    if processing_time < min_processing_time:
                        min_processing_time = processing_time
                    if execution_time < min_execution_time:
                        min_execution_time = execution_time

            min_times[(i,j)]["processing"] = min_processing_time
            min_times[(i,j)]["execution"] = min_execution_time

    # front
    for i in jobs:
        current_timestep = 0
        for j in jobs[i]:
            valid_interval_start = current_timestep + min_setup_duration

            valid_intervals[i][j]["start"] = valid_interval_start
            current_timestep += min_times[(i,j)]["execution"]

    # back
    for i in jobs:
        current_timestep = timesteps
        for j in reversed(jobs[i]):
            valid_interval_end = current_timestep - min_times[(i,j)]["processing"] + 1

            valid_intervals[i][j]["end"] = valid_interval_end
            current_timestep -= min_times[(i,j)]["execution"]

    return valid_intervals

def total_variable_count(data):
    machines = data["machines"]
    jobs = data["jobs"]
    timesteps = data["timesteps"]
    total_operations = 0
    total_speeds = 0
    total_machines = len(machines)
    for i in jobs:
        total_operations += len(jobs[i])
    for m in machines:
        total_speeds += len(machines[m]["speeds"])

    total_variables = 0
    total_variables += total_operations * total_speeds * timesteps # Operations
    total_variables += total_machines * (2 * timesteps + 1) # Machine turnon/-off
    total_variables += total_machines * timesteps # Maintenance
    total_variables += timesteps + 1 # Facility

    return total_variables

def createModel(data):
    start_time = time.time()
    print("creating model...")

    if data["model_type"] == "bqm":
        model = BinaryQuadraticModel('BINARY')
        bqm.addNecessaryConstraints(model, data)
        # import copy
        # model_copy = copy.deepcopy(model)
        # check1 = model_copy == model

        if data["energy_reciprocal"] > 0:
            bqm.addEnergyObjective(model, data, 1/data["energy_reciprocal"])
        if data["energy_cost_reciprocal"] > 0:
            bqm.addEnergyCostObjective(model, data, 1/data["energy_cost_reciprocal"])
        if data["power_reciprocal"] > 0:
            bqm.addPowerDiffObjective(model, data, 1/data["power_reciprocal"])

        # check2 = model_copy == model
        # if check1 and not check2:
        #     print("model has changed.")
        # else:
        #     print("model contains only necessary constraints.")

        total_variables = total_variable_count(data)
        data["variables_count"] = len(model.variables)
        data["pruned_variables"] = total_variables - len(model.variables)
        print(f"variables: {total_variables} -> {len(model.variables)}")

    elif data["model_type"] == "cqm":
        model = ConstrainedQuadraticModel()
        cqm.addNecessaryConstraints(model, data)
        if data["energy_reciprocal"] > 0:
            cqm.addEnergyObjective(model, data)
        model = lp.dumps(model)

    else:
        print("invalid model type")
        exit(1)

    end_time = time.time()
    print(f"model created in {end_time - start_time:.2f}s")

    return model

def main():
    arguments = sys.argv[1:]

    initial_values = {
        "total_operations": 20,
        "job_operations": 5,
        "total_machines": 4,
        "additional_speeds": 0,
        "max_setup_duration": 2,
        "max_maintenance_duration": 3,
        "timesteps": 30,
        "buffer_time": 5,
        "facility_power": 0,
        "model_type": "",
        "energy_reciprocal": 0,
        "energy_cost_reciprocal": 0,
        "power_reciprocal": 0
    }

    flag_mappings = {
        "o": "total_operations",
        "b": "job_operations",
        "m": "total_machines",
        "v": "additional_speeds",
        "sm": "max_setup_duration",
        "mm": "max_maintenance_duration",
        "t": "timesteps",
        "tb": "buffer_time",
        "f": "facility_power",
        "e": "energy_reciprocal",
        "c": "energy_cost_reciprocal",
        "p": "power_reciprocal",
    }

    FLAG = 0
    VALUE = 1
    expecting = FLAG
    current_flag = ""

    if len(arguments) > 0 and arguments[0] in ["help", "-?", "--help"]:
        print(USAGE)
        return

    flags = flag_mappings.keys()
    for arg in arguments:
        if expecting == FLAG:
            if arg[0] != "-":
                print(f"invalid argument '{arg}'")
                return
            current_flag = arg[1:]
            if current_flag in ["bqm", "cqm"]:
                initial_values["model_type"] = current_flag
                continue
            if current_flag not in flags:
                print(f"invalid flag '{arg}'")
                return
            expecting = VALUE
        else:
            value = 0
            try:
                value = int(arg)
            except:
                print(f"invalid value '{arg}'")
                return
            for flag in flags:
                if current_flag == flag:
                    initial_values[flag_mappings[flag]] = value
                    break
            expecting = FLAG

    total_operations = initial_values["total_operations"]
    job_operations = initial_values["job_operations"]
    total_machines = initial_values["total_machines"]
    additional_speeds = initial_values["additional_speeds"]
    max_setup_duration = initial_values["max_setup_duration"]
    max_maintenance_duration = initial_values["max_maintenance_duration"]
    timesteps = initial_values["timesteps"]
    buffer_time = initial_values["buffer_time"]
    facility_power = initial_values["facility_power"]
    energy_reciprocal = initial_values["energy_reciprocal"]
    energy_cost_reciprocal = initial_values["energy_cost_reciprocal"]
    power_reciprocal = initial_values["power_reciprocal"]
    model_type = initial_values["model_type"]

    if model_type == "":
        print("no model type specified")
        return

    print(f"total_operations: {total_operations}")
    print(f"job_operations: {job_operations}")
    print(f"total_machines: {total_machines}")
    print(f"additional_speeds: {additional_speeds}")
    print(f"max_setup_duration: {max_setup_duration}")
    print(f"max_maintenance_duration: {max_maintenance_duration}")
    print(f"timesteps: {timesteps}")
    print(f"buffer_time: {buffer_time}")
    print(f"facility_power: {facility_power}")
    print(f"energy_reciprocal: {energy_reciprocal}")
    print(f"energy_cost_reciprocal: {energy_cost_reciprocal}")
    print(f"power_reciprocal: {power_reciprocal}")
    print(f"model_type: {model_type}")
    print("--------------------")

    reserved_time = max(0, timesteps - buffer_time)
    jobs, machines = createRandomData(
        total_operations=total_operations,
        job_operations=job_operations,
        total_machines=total_machines,
        additional_speeds=additional_speeds,
        max_setup_duration=max_setup_duration,
        max_maintenance_duration=max_maintenance_duration,
        timesteps=reserved_time)
    if jobs == None or machines == None:
        exit(1)
    processing_durations = calculate_op_durations(jobs, machines)

    data = {}
    data["jobs"] = jobs
    data["machines"] = machines
    data["timesteps"] = timesteps
    data["facility_power"] = facility_power
    data["processing_durations"] = processing_durations
    data["model_type"] = model_type
    data["energy_reciprocal"] = energy_reciprocal
    data["energy_cost_reciprocal"] = energy_cost_reciprocal
    data["power_reciprocal"] = power_reciprocal
    data["valid_intervals"] = get_valid_intervals(data)

    model = createModel(data)
    output = {
        "model": model,
        "data": data
    }

    file_name = f"{model_type}_{total_operations}x{total_machines+additional_speeds}({total_machines})x{timesteps}({reserved_time})"

    if energy_reciprocal > 0 or power_reciprocal > 0 or energy_cost_reciprocal > 0:
        file_name += "_"
    if energy_reciprocal:
        file_name += f"e{energy_reciprocal}"
    if energy_cost_reciprocal:
        file_name += f"c{energy_cost_reciprocal}"
    if power_reciprocal:
        file_name += f"p{power_reciprocal}"
    file_name += ".pickle"

    cwd = os.path.dirname(os.path.realpath(__file__))
    dirs = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]
    if not "models" in dirs:
        os.mkdir(os.path.join(cwd, "models"))

    with open(os.path.join(cwd, "models", file_name), "wb") as file:
        pickle.dump(output, file)
        file.seek(0, os.SEEK_END)
        size = file.tell()

    orders = ["bytes", "KB", "MB", "GB", "TB"]
    order = 0

    while size > 1024 and order < len(orders) - 1:
        order += 1
        size /= 1024

    print(f"model size: {size:.2f} {orders[order]}")
    print(f"model name: {file_name}")
    # print(f"jobs = {json.dumps(jobs, indent=4)}")
    # print(f"machines = {json.dumps(machines, indent=4)}")



if __name__ == "__main__":
    main()