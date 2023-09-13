import dimod
from dimod import ConstrainedQuadraticModel, Binary, Integer
from dimod import ExactCQMSolver
from dwave.system import LeapHybridCQMSampler
import json
import itertools
from itertools import product
from math import ceil, log2, lcm
import numpy as np
import time
from random import randint
from compute.cqm_c.cqm_sampler import sample_exact_from_random, sample_genetic

def preprocess_jobs(jobs, machines):
    result = {}

    for i in jobs:
        for j in jobs[i]:
            steps = jobs[i][j]["steps"]

            for m in machines:
                speeds = machines[m]["speeds"]
                for v in speeds:
                    speed = speeds[v]["speed"]
                    duration = ceil(steps / speed)
                    result[(i,j,m,v)] = duration
    return result

def x(job, operation, machine, speed):
    return f"x.{job}.{operation}.{machine}.{speed}"

def start(job, operation):
    return f"start.{job}.{operation}"

def maint(machine):
    return f"maintenance.{machine}"

def machine_on(machine):
    return f"turnon.{machine}"

def machine_off(machine):
    return f"turnoff.{machine}"

def facility_off():
    return "facility"

def job_operation_pairs(jobs):
    pairs = [
        (i,j)
        for i in jobs for j in jobs[i]
    ]
    return pairs



def addProcessingConstraint(model: ConstrainedQuadraticModel, data):
    print("addProcessingConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for i, j in job_operation_pairs(jobs):
        variables = [
            Binary(x(i,j,m,v))
            for m in machines for v in machines[m]["speeds"]
        ]
        model.add_constraint_from_comparison(sum(variables) == 1, label=f"processing.{i}.{j}")

        durations = [
            (Binary(x(i,j,m,v)), processing_durations[(i,j,m,v)])
            for m in machines for v in machines[m]["speeds"]
        ]
        duration_ij = sum([var * duration for var, duration in durations])
        start_ij = Integer(start(i,j),
                           lower_bound=valid_intervals[i][j]["start"],
                           upper_bound=valid_intervals[i][j]["end"]-1)

        model.add_constraint_from_comparison(start_ij + duration_ij <= timesteps, label=f"makespan.{i}.{j}")
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addOverlappingConstraint(model: ConstrainedQuadraticModel, data):
    print("addOverlappingConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for i, j in job_operation_pairs(jobs):
        setups = [
            (Binary(x(i,j,m,v)), machines[m]["setup_duration"])
            for m in machines for v in machines[m]["speeds"]
        ]
        durations = [
            (Binary(x(i,j,m,v)), processing_durations[(i,j,m,v)])
            for m in machines for v in machines[m]["speeds"]
        ]
        setup_ij = sum([var * duration for var, duration in setups])
        start_ij = Integer(start(i,j),
                           lower_bound=valid_intervals[i][j]["start"],
                           upper_bound=valid_intervals[i][j]["end"]-1)
        duration_ij = sum([var * duration for var, duration in durations])

        model.add_constraint_from_comparison(start_ij - setup_ij >= 0, label=f"setup.{i}.{j}")

        for i2, j2 in job_operation_pairs(jobs):
            if i == i2 and j == j2:
                continue

            setups = [
                (Binary(x(i2,j2,m,v)), machines[m]["setup_duration"])
                for m in machines for v in machines[m]["speeds"]
            ]
            durations = [
                (Binary(x(i2,j2,m,v)), processing_durations[(i2,j2,m,v)])
                for m in machines for v in machines[m]["speeds"]
            ]
            setup_ij2 = sum([var * duration for var, duration in setups])
            start_ij2 = Integer(start(i2,j2),
                                lower_bound=valid_intervals[i2][j2]["start"],
                                upper_bound=valid_intervals[i2][j2]["end"]-1)
            duration_ij2 = sum([var * duration for var, duration in durations])

            same_machine = 0
            for m in machines:
                speeds = machines[m]["speeds"]
                for v, v2 in product(speeds, speeds):
                    same_machine += Binary(x(i,j,m,v)) * Binary(x(i2,j2,m,v2))
            threshold = (1 - same_machine) * timesteps**2

            model.add_constraint_from_comparison(- threshold + (start_ij - setup_ij - start_ij2 - duration_ij2) * (start_ij2 - setup_ij2 - start_ij - duration_ij) <= 0, label=f"overlapping.{i}.{j}.{i2}.{j2}")
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addSequenceConstraint(model: ConstrainedQuadraticModel, data):
    print("addSequenceConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for i in jobs:
        operation_indices = list(jobs[i].keys())
        for op_index in range(1, len(operation_indices)):
            j = operation_indices[op_index]
            j_prev = operation_indices[op_index - 1]

            durations = [
                (Binary(x(i,j_prev,m,v)), processing_durations[(i,j_prev,m,v)])
                for m in machines for v in machines[m]["speeds"]
            ]
            duration_prev = sum([var * duration for var, duration in durations])

            setups = [
                (Binary(x(i,j,m,v)), machines[m]["setup_duration"])
                for m in machines for v in machines[m]["speeds"]
            ]
            setup_current = sum([var * duration for var, duration in setups])

            start_j = Integer(start(i,j),
                              lower_bound=valid_intervals[i][j]["start"],
                              upper_bound=valid_intervals[i][j]["end"]-1)
            start_j_prev = Integer(start(i,j_prev),
                                   lower_bound=valid_intervals[i][j_prev]["start"],
                                   upper_bound=valid_intervals[i][j_prev]["end"]-1)

            model.add_constraint_from_comparison(start_j - start_j_prev - duration_prev - setup_current >= 0, label=f"sequence.{i}.{j_prev}.{j}")
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addMaintenanceConstraint(model: ConstrainedQuadraticModel, data):
    print("addMaintenanceConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for m in machines:
        setup_duration = machines[m]["setup_duration"]
        maintenance_duration = machines[m]["maintenance_duration"]
        maintenance_start = Integer(f"maintenance.{m}", upper_bound=timesteps-maintenance_duration)

        for i, j in job_operation_pairs(jobs):
            durations = [
                (Binary(x(i,j,m,v)), processing_durations[(i,j,m,v)])
                for v in machines[m]["speeds"]
            ]
            duration_ij = sum([var * duration for var, duration in durations])
            start_ij = Integer(start(i,j),
                               lower_bound=valid_intervals[i][j]["start"],
                               upper_bound=valid_intervals[i][j]["end"]-1)

            same_machine = 0
            for v in machines[m]["speeds"]:
                same_machine += Binary(x(i,j,m,v))
            threshold = (1 - same_machine) * timesteps**2

            model.add_constraint_from_comparison(-threshold + (maintenance_start - start_ij - duration_ij) * (start_ij - setup_duration - maintenance_start - maintenance_duration) <= 0, label=f"maintenance.{m}.{i}.{j}")
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addMachineOffConstraint(model: ConstrainedQuadraticModel, data):
    print("addMachineOffConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for m in machines:
        setup_duration = machines[m]["setup_duration"]
        maintenance_duration = machines[m]["maintenance_duration"]

        turnon = Integer(machine_on(m), upper_bound=timesteps-1)
        for i, j in job_operation_pairs(jobs):
            starts = [
                (Binary(x(i,j,m,v)), Integer(start(i,j),
                                             lower_bound=valid_intervals[i][j]["start"],
                                             upper_bound=valid_intervals[i][j]["end"]-1))
                for v in machines[m]["speeds"]
            ]
            start_ij = sum([var * start_time for var, start_time in starts])

            same_machine = 0
            for v in machines[m]["speeds"]:
                same_machine += Binary(x(i,j,m,v))
            threshold = (1 - same_machine) * 2 * timesteps

            model.add_constraint_from_comparison(threshold + start_ij - setup_duration - turnon >= 0, label=f"turnon.{m}.{i}.{j}")

        turnoff = Integer(machine_off(m), upper_bound=timesteps)
        for i, j in job_operation_pairs(jobs):
            durations = [
                (Binary(x(i,j,m,v)), processing_durations[(i,j,m,v)])
                for v in machines[m]["speeds"]
            ]
            starts = [
                (Binary(x(i,j,m,v)), Integer(start(i,j),
                                             lower_bound=valid_intervals[i][j]["start"],
                                             upper_bound=valid_intervals[i][j]["end"]-1))
                for v in machines[m]["speeds"]
            ]
            start_ij = sum([var * start_time for var, start_time in starts])
            duration_ij = sum([var * duration for var, duration in durations])

            model.add_constraint_from_comparison(turnoff - start_ij - duration_ij >= 0, label=f"turnoff.{m}.{i}.{j}")

        maintenance_start = Integer(maint(m), upper_bound=timesteps-maintenance_duration)
        model.add_constraint_from_comparison(maintenance_start - turnon >= 0, label=f"turnon.{m}.maintenance")
        model.add_constraint_from_comparison(turnoff - maintenance_start - maintenance_duration >= 0, label=f"turnoff.{m}.maintenance")
        model.add_constraint_from_comparison(turnoff - turnon >= 0, label=f"turnoff.{m}.turnon")
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addFacilityConstraint(model: ConstrainedQuadraticModel, data):
    print("addFacilityConstraint... ", end="")
    start_time = time.time()

    machines = data["machines"]
    timesteps = data["timesteps"]

    facility = Integer(facility_off(), upper_bound=timesteps)
    for m in machines:
        turnoff = Integer(machine_off(m), upper_bound=timesteps)
        model.add_constraint_from_comparison(facility - turnoff >= 0, label=f"facility.{m}.turnoff")
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addNecessaryConstraints(model: ConstrainedQuadraticModel, data):
    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = preprocess_jobs(jobs, machines)

    addProcessingConstraint(model, data)
    addOverlappingConstraint(model, data)
    addSequenceConstraint(model, data)
    addMaintenanceConstraint(model, data)
    addMachineOffConstraint(model, data)
    addFacilityConstraint(model, data)

def addEnergyObjective(model: ConstrainedQuadraticModel, data):
    print("addEnergyObjective... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    facility_power = data["facility_power"]
    processing_durations = preprocess_jobs(jobs, machines)

    total_energy = 0
    for m in machines:
        idle_power = machines[m]["idle_power"]
        maintenace_power = machines[m]["maintenance_power"]
        maintenace_duration = machines[m]["maintenance_duration"]
        setup_power = machines[m]["setup_power"]
        setup_duration = machines[m]["setup_duration"]
        speeds = machines[m]["speeds"]

        total_energy += maintenace_duration * (maintenace_power - idle_power)

        for v in speeds:
            processing_power = speeds[v]["processing_power"]
            efficiency = speeds[v]["efficiency"]
            for i, j in job_operation_pairs(jobs):
                total_energy += Binary(x(i,j,m,v)) * setup_duration * (setup_power - idle_power)

                energy = Binary(x(i,j,m,v)) * jobs[i][j]["energy"]
                duration = Binary(x(i,j,m,v)) * processing_durations[(i,j,m,v)]
                total_energy += duration * (processing_power - idle_power)
                total_energy += energy / efficiency
        total_energy += timesteps * idle_power

        turnon = Integer(machine_on(m), upper_bound=timesteps-1)
        turnoff = Integer(machine_off(m), upper_bound=timesteps)
        total_energy -= turnon * idle_power
        total_energy -= (timesteps - turnoff) * idle_power

    facility = Integer("facility", upper_bound=timesteps)
    total_energy += facility_power * facility

    model.set_objective(total_energy)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")


def serialize(model: ConstrainedQuadraticModel):
    print("serialize... ", end="")
    start_time = time.time()

    serialized = {}
    serialized["variables"] = {}
    serialized["constraints"] = {}
    serialized["objective"] = {}

    variables = {}
    for var in model.variables:
        variables[var] = {
            "type": model.vartype(var).name,
            "lower_bound": model.lower_bound(var),
            "upper_bound": model.upper_bound(var)
        }
    serialized["variables"] = variables

    for constraint in model.constraints:
        equation = model.constraints[constraint]
        lhs = equation.lhs
        rhs = equation.rhs - lhs.offset
        sense = equation.sense
        linear = dict(lhs.linear)
        quadratic = dict(lhs.quadratic)

        sense = sense.name
        rhs = float(rhs)

        constraint_serialized = {
            "linear": linear,
            "quadratic": quadratic,
            "sense": sense,
            "rhs": rhs
        }
        serialized["constraints"][constraint] = constraint_serialized

    linear = dict(model.objective.linear)
    quadratic = dict(model.objective.quadratic)
    serialized["objective"]["linear"] = linear
    serialized["objective"]["quadratic"] = quadratic

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")
    return serialized

def deserialize(serialized):
    print("deserialize... ", end="")
    start_time = time.time()

    model = ConstrainedQuadraticModel()

    variables = serialized["variables"]
    for var in variables:
        type = variables[var]["type"]
        lower_bound = variables[var]["lower_bound"]
        upper_bound = variables[var]["upper_bound"]

        model.add_variable(type, var, lower_bound=lower_bound, upper_bound=upper_bound)

    constraints = serialized["constraints"]
    for constraint in constraints:
        linear = constraints[constraint]["linear"]
        quadratic = constraints[constraint]["quadratic"]
        sense = constraints[constraint]["sense"]
        rhs = constraints[constraint]["rhs"]

        linear_sum = 0
        for var in linear:
            type = variables[var]["type"]
            if type == "BINARY":
                model_variable = Binary(var)
            elif type == "INTEGER":
                lower_bound = variables[var]["lower_bound"]
                upper_bound = variables[var]["upper_bound"]
                model_variable = Integer(var, lower_bound=lower_bound, upper_bound=upper_bound)
            coefficient = linear[var]
            linear_sum += coefficient * model_variable
        quadratic_sum = 0
        for (var1, var2) in quadratic:
            type1 = variables[var1]["type"]
            type2 = variables[var2]["type"]
            model_variable1 = None
            model_variable2 = None
            if type1 == "BINARY":
                model_variable1 = Binary(var1)
            elif type1 == "INTEGER":
                lower_bound = variables[var1]["lower_bound"]
                upper_bound = variables[var1]["upper_bound"]
                model_variable1 = Integer(var1, lower_bound=lower_bound, upper_bound=upper_bound)
            if type2 == "BINARY":
                model_variable2 = Binary(var2)
            elif type2 == "INTEGER":
                lower_bound = variables[var2]["lower_bound"]
                upper_bound = variables[var2]["upper_bound"]
                model_variable2 = Integer(var2, lower_bound=lower_bound, upper_bound=upper_bound)
            coefficient = quadratic[(var1, var2)]
            quadratic_sum += coefficient * model_variable1 * model_variable2

        if sense == "Le":
            model.add_constraint_from_comparison(linear_sum + quadratic_sum <= rhs, label=constraint)
        elif sense == "Ge":
            model.add_constraint_from_comparison(linear_sum + quadratic_sum >= rhs, label=constraint)
        elif sense == "Eq":
            model.add_constraint_from_comparison(linear_sum + quadratic_sum == rhs, label=constraint)

    objective = serialized["objective"]
    linear = objective["linear"]
    quadratic = objective["quadratic"]
    linear_sum = 0
    for var in linear:
        type = variables[var]["type"]
        if type == "BINARY":
            model_variable = Binary(var)
        elif type == "INTEGER":
            lower_bound = variables[var]["lower_bound"]
            upper_bound = variables[var]["upper_bound"]
            model_variable = Integer(var, lower_bound=lower_bound, upper_bound=upper_bound)
        coefficient = linear[var]
        linear_sum += coefficient * model_variable
    quadratic_sum = 0
    for (var1, var2) in quadratic:
        type1 = variables[var1]["type"]
        type2 = variables[var2]["type"]
        model_variable1 = None
        model_variable2 = None
        if type1 == "BINARY":
            model_variable1 = Binary(var1)
        elif type1 == "INTEGER":
            lower_bound = variables[var1]["lower_bound"]
            upper_bound = variables[var1]["upper_bound"]
            model_variable1 = Integer(var1, lower_bound=lower_bound, upper_bound=upper_bound)
        if type2 == "BINARY":
            model_variable2 = Binary(var2)
        elif type2 == "INTEGER":
            lower_bound = variables[var2]["lower_bound"]
            upper_bound = variables[var2]["upper_bound"]
            model_variable2 = Integer(var2, lower_bound=lower_bound, upper_bound=upper_bound)
        coefficient = quadratic[(var1, var2)]
        quadratic_sum += coefficient * model_variable1 * model_variable2
    model.set_objective(linear_sum + quadratic_sum)

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")
    return model


def getEnergyValue(data, sample):
    print("getEnergyValue... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    facility_power = data["facility_power"]
    processing_durations = data["processing_durations"]

    total_energy = 0
    for m in machines:
        idle_power = machines[m]["idle_power"]
        maintenace_power = machines[m]["maintenance_power"]
        maintenace_duration = machines[m]["maintenance_duration"]
        setup_power = machines[m]["setup_power"]
        setup_duration = machines[m]["setup_duration"]
        speeds = machines[m]["speeds"]

        total_energy += maintenace_duration * (maintenace_power - idle_power)

        for v in speeds:
            processing_power = speeds[v]["processing_power"]
            efficiency = speeds[v]["efficiency"]
            for i, j in job_operation_pairs(jobs):
                if sample.get(x(i,j,m,v), 0) == 0:
                    continue
                duration = processing_durations[(i,j,m,v)]
                energy = jobs[i][j]["energy"]

                total_energy += setup_duration * (setup_power - idle_power)
                total_energy += duration * (processing_power - idle_power)
                total_energy += energy / efficiency
        total_energy += timesteps * idle_power

        turnon = sample[machine_on(m)]
        turnoff = sample[machine_off(m)]

        total_energy -= turnon * idle_power
        total_energy -= (timesteps - turnoff) * idle_power

    facility = sample[facility_off()]
    total_energy += facility_power * facility

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")
    return total_energy



def useExact(model: ConstrainedQuadraticModel, sampleset_size) -> list:
    sampleset = sample_exact_from_random(model, sampleset_size)
    return sampleset

def useGenetic(model: ConstrainedQuadraticModel, sampleset_size) -> list:
    print("Using genetic algorithm")
    sampleset = sample_genetic(model, sampleset_size)
    return sampleset

def useLeap(model: ConstrainedQuadraticModel, sampleset_size, label) -> list:
    sampler = LeapHybridCQMSampler()

    sampleset = sampler.sample_cqm(model, label = f"{label} LEAP")

    print(json.dumps(sampleset.info, indent=4))

    sampleset = list(sampleset)
    for i in range(len(sampleset)):
        sample = sampleset[i]
        sample = {k: int(v) for k, v in sample.items()}
        sampleset[i] = sample

    return sampleset
