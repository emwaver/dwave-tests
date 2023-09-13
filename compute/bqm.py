from dimod import BinaryQuadraticModel
from dimod import SimulatedAnnealingSampler
from dwave.samplers import TabuSampler
from dwave.system import LeapHybridBQMSampler
from random import randint
import json
import time
import itertools
from itertools import product
from math import ceil, log2, lcm
from math import sin, pi

def estimateEnergyConsumption(jobs, machines, timesteps, processing_durations):

    '''
        Zunächst werden alle Operationen, Steps und Energien aller Jobs aufsummiert.

        Dann wird berechnet, wie lange eine Maschine für alle Operationen zusammen benötigen würde.
        Diese Zeit setzt sich aus der erwarteten Bearbeitungszeit für alle Operationen und der Setup-Zeit für alle Operationen zusammen.
        Da eine Maschine i.d.R. nicht alle Operationen bearbeiten kann, wird daraus die durchschnittliche Anzahl an Operationen ermittelt,
        die einer Maschine zugewiesen werden, sollte sie voll belegt sein.
        Der daraus resultierende Wert wird im Folgenden als "durchschnittliche Maschnenkapazität" bezeichnet.

        Da in der Maschinenbelegungsplanung nicht alle Maschinen voll ausgelastet sind, wird die durchschnittliche Maschinenkapauität mit der
        Gesamtkapazität aller Maschinen normiert. Für jede Maschine ist nun die zu erwartende durchschnittliche Anzahl an Operationen bekannt.

        Mit dieser Information kann nun die erwartete Bearbeitungszeit und alle Energien für jede Maschine berechnet werden.

        Die geschätzte Gesamtenergie unterschätzt die tatsächliche Gesamteenergie oft nur um wenige Prozent.
    '''

    total_operations = 0
    total_steps = 0
    total_energy = 0
    for i in jobs:
        for j in jobs[i]:
            total_operations += 1
            total_steps += jobs[i][j]["steps"]
            total_energy += jobs[i][j]["energy"]

    av_machine_capacity = {}
    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        setup_duration = machines[m]["setup_duration"]
        available_time = timesteps - maintenance_duration

        speeds = machines[m]["speeds"]

        all_operations_processing_time = 0
        for i in jobs:
            for j in jobs[i]:
                av_duration = 0
                for v in speeds:
                    av_duration += processing_durations[(i,j,m,v)]
                av_duration /= len(speeds)
                all_operations_processing_time += av_duration

        all_operations_processing_time += setup_duration * total_operations
        portion = available_time / all_operations_processing_time
        av_machine_capacity[m] = portion * total_operations

    total_machine_capacity = sum([av_machine_capacity[m] for m in av_machine_capacity])
    for c in av_machine_capacity:
        av_machine_capacity[c] *= total_operations / total_machine_capacity

    estimated_total_energy = 0
    for m in machines:
        idle_power = machines[m]["idle_power"]
        maintenance_power = machines[m]["maintenance_power"]
        maintenance_duration = machines[m]["maintenance_duration"]
        setup_power = machines[m]["setup_power"]
        setup_duration = machines[m]["setup_duration"]
        processing_power = sum([machines[m]["speeds"][v]["processing_power"] for v in machines[m]["speeds"]]) / len(machines[m]["speeds"])

        speeds = machines[m]["speeds"]

        estimated_processing_time = 0
        estimated_processing_energy = 0
        for i in jobs:
            for j in jobs[i]:
                av_duration = 0
                av_energy = 0
                for v in speeds:
                    av_duration += processing_durations[(i,j,m,v)]
                    av_energy += jobs[i][j]["energy"] / speeds[v]["efficiency"]
                av_duration /= len(speeds)
                av_energy /= len(speeds)
                estimated_processing_time += av_duration
                estimated_processing_energy += av_energy

        portion = av_machine_capacity[m] / total_operations
        estimated_processing_time *= portion
        estimated_processing_energy = estimated_processing_energy * portion
        estimated_processing_energy += estimated_processing_time * processing_power


        estimated_setup_time = setup_duration * av_machine_capacity[m]
        estimated_setup_energy = setup_power * estimated_setup_time

        estimated_maintenance_energy = maintenance_power * maintenance_duration

        estimated_idle_time = timesteps - estimated_processing_time - estimated_setup_time - maintenance_duration
        estimated_idle_energy = idle_power * estimated_idle_time

        estimated_total_energy += estimated_idle_energy + estimated_maintenance_energy + estimated_setup_energy + estimated_processing_energy

    return estimated_total_energy

def x(job, operation, machine, timestep, speed):
    return f"x.{job}.{operation}.{machine}.{timestep}.{speed}"

def maintenance(machine, timestep):
    return f"maintenance.{machine}.{timestep}"

def turnon(machine, timestep):
    return f"turnon.{machine}.{timestep}"

def turnoff(machine, timestep):
    return f"turnoff.{machine}.{timestep}"

def facility(timestep):
    return f"facility_off.{timestep}"

def job_operation_pairs(jobs):
    pairs = [
        (i,j)
        for i in jobs for j in jobs[i]
    ]
    return pairs

def machine_speed_pairs(machines):
    pairs = [
        (m,v)
        for m in machines for v in machines[m]["speeds"]
    ]
    return pairs



def addProcessingConstraint(model: BinaryQuadraticModel, weight, data):
    print("addProcessingConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for i, j in job_operation_pairs(jobs):

        # jeder Operation muss genau 1 Mal gestartet werden
        variables = [
            (x(i,j,m,t,v), 1)
            for m, v in machine_speed_pairs(machines) for t in range(timesteps)
            if valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]
        ]

        model.add_linear_equality_constraint(terms = variables,
                                           lagrange_multiplier = weight,
                                           constant = -1)

        # Operationen dürfen nicht zu spät starten
        for m, v in machine_speed_pairs(machines):
            duration = processing_durations[(i,j,m,v)]
            for t in range(timesteps - duration + 1, timesteps):
                if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                    continue
                model.add_linear(v = x(i,j,m,t,v),
                               bias = weight)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addOverlappingConstraint(model: BinaryQuadraticModel, weight, data):
    print("addOverlappingConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    # zu keinem Zeitpuntk darf irgendeine Maschine doppelt belegt sein
    for m in machines:
        for i, i2 in product(jobs, repeat=2):
            for j, j2 in product(jobs[i], jobs[i2]):
                if i == i2 and j == j2:
                    continue

                speeds = machines[m]["speeds"]
                for v, v2 in product(speeds, repeat=2):
                    duration = processing_durations[(i,j,m,v)]
                    for t, t2 in product(range(timesteps), repeat=2):
                        if not (0 <= t2 - t < duration):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                            continue
                        if not (valid_intervals[i2][j2]["start"] <= t2 < valid_intervals[i2][j2]["end"]):
                            continue
                        model.add_quadratic(u = x(i,j,m,t,v),
                                        v = x(i2,j2,m,t2,v2),
                                        bias = weight)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addSequenceConstraint(model: BinaryQuadraticModel, weight, data):
    print("addSequenceConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    # Operation innerhalb eines Auftrags müssen in der richtigen Reihenfolge bearbeitet werden
    for i in jobs:
        for j, j2 in product(jobs[i], repeat=2):
            for m, v in machine_speed_pairs(machines):
                duration = processing_durations[(i,j,m,v)]
                for t in range(timesteps - duration + 1):
                    for m2, v2 in machine_speed_pairs(machines):
                        for t2 in range(t + duration):
                            if not (j2 > j):
                                continue
                            if not (m2 != m or t2 < t):
                                continue
                            if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                                continue
                            if not (valid_intervals[i][j2]["start"] <= t2 < valid_intervals[i][j2]["end"]):
                                continue
                            model.add_quadratic(u = x(i,j,m,t,v),
                                              v = x(i,j2,m2,t2,v2),
                                              bias = weight)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addSetupConstraint(model: BinaryQuadraticModel, weight, data):
    print("addSetupConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for m in machines:
        setup_duration = machines[m]["setup_duration"]
        speeds = machines[m]["speeds"]

        if setup_duration == 0:
            continue

        # Operationen dürfen nicht zu früh starten
        for t in range(setup_duration):
            for i, j in job_operation_pairs(jobs):
                for v in speeds:
                    if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                        continue
                    model.add_linear(v = x(i,j,m,t,v),
                                   bias = weight)


        # Während einer Operation darf keine Setup Zeit einer anderen Operation stattfinden
        for t2 in range(setup_duration, timesteps):
            for (i, j), (i2, j2) in product(job_operation_pairs(jobs), repeat=2):
                for v, v2 in product(speeds, repeat=2):
                    duration = processing_durations[(i,j,m,v)]
                    for t in range(t2):
                        if not (t2 - t < setup_duration + duration):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                            continue
                        if not (valid_intervals[i2][j2]["start"] <= t2 < valid_intervals[i2][j2]["end"]):
                            continue
                        model.add_quadratic(u = x(i,j,m,t,v),
                                          v = x(i2,j2,m,t2,v2),
                                          bias = weight)

    # Wenn ein Auftrag die Maschine wechselt, darf die Setup Zeit nicht die vorangehende Operation zeitlich überlagern
    for i in jobs:
        for j, j2 in product(jobs[i], repeat=2):
            if not (j2 > j):
                continue
            for m, m2 in product(machines, repeat=2):
                speeds = machines[m]["speeds"]
                speeds2 = machines[m2]["speeds"]
                for v, v2 in product(speeds, speeds2):
                    duration = processing_durations[(i,j,m,v)]
                    setup_duration2 = machines[m2]["setup_duration"]
                    for t, t2 in product(range(timesteps), repeat=2):
                        if not (t2 - t < duration + setup_duration2):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                            continue
                        if not (valid_intervals[i][j2]["start"] <= t2 < valid_intervals[i][j2]["end"]):
                            continue
                        model.add_quadratic(u = x(i,j,m,t,v), v = x(i,j2,m2,t2,v2), bias = weight)

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addMaintenanceConstraint(model: BinaryQuadraticModel, weight, data):
    print("addMaintenanceConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        setup_duration = machines[m]["setup_duration"]
        if maintenance_duration == 0:
            continue

        # Die Wartungsphase muss genau einen Start haben
        variables = [
            (maintenance(m,t), 1)
            for t in range(timesteps)
        ]
        model.add_linear_equality_constraint(terms = variables, lagrange_multiplier = weight, constant = -1)

        # Dieser Start darf nicht zu spät sein
        for t in range(timesteps):
            if not (t > timesteps - maintenance_duration):
                continue
            model.add_linear(v = maintenance(m,t), bias = weight)

        # Wartungsphasen und Operationen dürfen sich nicht überlappen
        speeds = machines[m]["speeds"]
        for t in range(timesteps):
            for i, j in job_operation_pairs(jobs):
                for v in speeds:
                    duration = processing_durations[(i,j,m,v)]

                    # Innerhalb einer Operation/Setupphase darf keine Wartungsphase starten
                    for t2 in range(timesteps):
                        if not (-setup_duration <= t2 - t < duration):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                            continue
                        model.add_quadratic(u = x(i,j,m,t,v), v = maintenance(m,t2), bias = weight)

                    # Innerhalb einer Wartungsphase darf keine Operation/Setupphase starten
                    for t2 in range(timesteps):
                        if not (0 <=  t2 - t < maintenance_duration + setup_duration):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                            continue
                        model.add_quadratic(u = maintenance(m,t), v = x(i,j,m,t2,v), bias = weight)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addMachineOffConstraint(model: BinaryQuadraticModel, weight, data):
    print("addMachineOffConstraint... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    # turn on machine
    for m in machines:
        setup_duration = machines[m]["setup_duration"]
        speeds = machines[m]["speeds"]

        # Jede Maschine wird genau einmal eingeschaltet
        variables = [
            (turnon(m,t), 1)
            for t in range(timesteps)
        ]
        model.add_linear_equality_constraint(terms = variables, lagrange_multiplier = weight, constant = -1)

        # Vor dem Einschalten darf keine Wartungsphase starten
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t2 < t):
                continue
            model.add_quadratic(u = turnon(m,t), v = maintenance(m,t2), bias = weight)

        # Vor dem Einschalten darf keine Operation/Setupphase starten
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t2 < t + setup_duration):
                continue
            for i, j in job_operation_pairs(jobs):
                for v in speeds:
                    if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                        continue
                    model.add_quadratic(u = turnon(m,t), v = x(i,j,m,t2,v), bias = weight)

    # turn off machine
    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        speeds = machines[m]["speeds"]

        # Jede Maschine wird genau einmal ausgeschaltet
        variables = [
            (turnoff(m,t), 1)
            # Die Maschine muss bis zum Ende durchlaufen können
            # Da es einen Turnoff-Zeitpunkt geben muss, muss für timesteps+1 eine weitere Variable eingeführt werden
            # Die andere Option wäre, eine Slack-Variable einzuführen, um eine Ungleichungsbedingungs <= 1 zu schaffen
            for t in range(timesteps+1)
        ]
        model.add_linear_equality_constraint(terms = variables, lagrange_multiplier = weight, constant = -1)

        # Die Maschine darf erst nach abgeschlossenen Wartungsphasen ausgeschaltet werden
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t - t2 < maintenance_duration):
                continue
            model.add_quadratic(u = turnoff(m,t), v = maintenance(m,t2), bias = weight)

        # Die Maschine darf erst nach abgeschlossenen Operationen ausgeschaltet werden
        for t, t2 in product(range(timesteps), repeat=2):
            for i, j in job_operation_pairs(jobs):
                for v in speeds:
                    duration = processing_durations[(i,j,m,v)]
                    if not (t - t2 < duration):
                        continue
                    if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                        continue
                    model.add_quadratic(u = turnoff(m,t), v = x(i,j,m,t2,v), bias = weight)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addFacilityConstraint(model: BinaryQuadraticModel, weight, data):
    print("addFacilityConstraint... ", end="")
    start_time = time.time()

    machines = data["machines"]
    timesteps = data["timesteps"]

    variables = [
        (facility(t), 1)
        for t in range(timesteps+1)
    ]
    model.add_linear_equality_constraint(terms = variables, lagrange_multiplier = weight, constant = -1)

    for m in machines:
        for t, t2 in product(range(timesteps+1), repeat=2):
            if not (t2 > t):
                continue
            model.add_quadratic(u = facility(t), v = turnoff(m,t2), bias = weight)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addNecessaryConstraints(model: BinaryQuadraticModel, data):
    addProcessingConstraint(model, 1, data)
    addOverlappingConstraint(model, 1, data)
    addSequenceConstraint(model, 1, data)
    addSetupConstraint(model, 1, data)
    addMaintenanceConstraint(model, 1, data)
    addMachineOffConstraint(model, 1, data)
    addFacilityConstraint(model, 1, data)

def addEnergyObjective(model: BinaryQuadraticModel, data, weight):
    print("addEnergyObjective... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    facility_power = data["facility_power"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for m in machines:
        idle_power = machines[m]["idle_power"]
        maintenance_power = machines[m]["maintenance_power"]
        maintenance_duration = machines[m]["maintenance_duration"]
        setup_power = machines[m]["setup_power"]
        setup_duration = machines[m]["setup_duration"]

        model.offset += maintenance_duration * (maintenance_power - idle_power) * weight

        speeds = machines[m]["speeds"]
        for i, j in job_operation_pairs(jobs):
            for v in speeds:
                processing_power = speeds[v]["processing_power"]
                efficiency = speeds[v]["efficiency"]
                for t in range(timesteps):
                    duration = processing_durations[(i,j,m,v)]
                    energy = jobs[i][j]["energy"]

                    energy_bias = setup_duration * (setup_power - idle_power)
                    energy_bias += duration * (processing_power - idle_power)
                    energy_bias += energy / efficiency

                    if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                        continue
                    model.add_linear(v = x(i,j,m,t,v), bias = energy_bias * weight)

        model.offset += idle_power * timesteps * weight
        for t in range(timesteps):
            model.add_linear(v = turnon(m,t), bias = - t * idle_power * weight)
            model.add_linear(v = turnoff(m,t), bias = - (timesteps - t) * idle_power * weight)

    model.offset += timesteps * facility_power * weight
    for t in range(timesteps):
        model.add_linear(v = facility(t), bias = - (timesteps - t) * facility_power * weight)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addEnergyCostObjective(model: BinaryQuadraticModel, data, weight):
    print("addEnergyCostObjective... ", end="")
    start_time = time.time()

    timesteps = data["timesteps"]
    x_norm = [i/timesteps for i in range(timesteps+1)] + [1.0]
    x_price = [2*pi*x for x in x_norm]
    y_price = [0.5*sin(x)+1 for x in x_price]
    data["price"] = y_price

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    facility_power = data["facility_power"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]
    powers = [[] for _ in range(timesteps)]

    # Idle
    for m in machines:
        idle_power = machines[m]["idle_power"]
        model.offset += idle_power * timesteps * weight

    # Maintenance
    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        maintenance_power = machines[m]["maintenance_power"]
        idle_power = machines[m]["idle_power"]
        for t, t2 in product(range(timesteps), repeat=2):
            if not (0 <= t - t2 < maintenance_duration):
                continue
            powers[t].append((maintenance(m,t2), (maintenance_power - idle_power)))

    # Setup
    for m, v in machine_speed_pairs(machines):
        setup_duration = machines[m]["setup_duration"]
        setup_power = machines[m]["setup_power"]
        idle_power = machines[m]["idle_power"]
        for i, j in job_operation_pairs(jobs):
            for t, t2 in product(range(timesteps), repeat=2):
                if not (0 < t2 - t <= setup_duration):
                    continue
                if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                    continue
                powers[t].append((x(i,j,m,t2,v), (setup_power - idle_power)))

    # Processing
    for m, v in machine_speed_pairs(machines):
        idle_power = machines[m]["idle_power"]
        for i, j in job_operation_pairs(jobs):
            for t, t2 in product(range(timesteps), repeat=2):
                if not (0 <= t - t2 < processing_durations[(i,j,m,v)]):
                    continue

                processing_power = machines[m]["speeds"][v]["processing_power"]
                efficiency = machines[m]["speeds"][v]["efficiency"]
                energy = jobs[i][j]["energy"]
                duration = processing_durations[(i,j,m,v)]

                energy_bias = (processing_power)
                energy_bias += energy / duration / efficiency

                if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                    continue
                powers[t].append((x(i,j,m,t2,v), (energy_bias - idle_power)))

    # Off pre
    for m in machines:
        idle_power = machines[m]["idle_power"]
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t < t2):
                continue
            powers[t].append((turnon(m,t2), -idle_power))

    # Off post
    for m in machines:
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t2 <= t):
                continue

            powers[t].append((turnoff(m,t2), -idle_power))

    # Facility Power
    for t, t2 in product(range(timesteps), repeat=2):
        if not (t2 <= t):
            continue
        powers[t].append((facility(t2), -facility_power))

    # Price
    for powers, price in zip(powers, y_price):
        for variable, power in powers:
            model.add_linear(v = variable, bias = power * price * weight)

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

def addPowerDiffObjective(model: BinaryQuadraticModel, data, weight):
    print("addPowerDiffObjective... ", end="")
    start_time = time.time()

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    facility_power = data["facility_power"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    powers = [[] for _ in range(timesteps)]

    # Maintenance
    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        maintenance_power = machines[m]["maintenance_power"]
        idle_power = machines[m]["idle_power"]
        for t, t2 in product(range(timesteps), repeat=2):
            if not (0 <= t - t2 < maintenance_duration):
                continue
            powers[t].append((maintenance(m,t2), (maintenance_power - idle_power)))

    # Setup
    for m, v in machine_speed_pairs(machines):
        setup_duration = machines[m]["setup_duration"]
        setup_power = machines[m]["setup_power"]
        idle_power = machines[m]["idle_power"]
        for i, j in job_operation_pairs(jobs):
            for t, t2 in product(range(timesteps), repeat=2):
                if not (0 < t2 - t <= setup_duration):
                    continue
                if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                    continue
                powers[t].append((x(i,j,m,t2,v), (setup_power - idle_power)))

    # Processing
    for m, v in machine_speed_pairs(machines):
        idle_power = machines[m]["idle_power"]
        for i, j in job_operation_pairs(jobs):
            for t, t2 in product(range(timesteps), repeat=2):
                if not (0 <= t - t2 < processing_durations[(i,j,m,v)]):
                    continue

                processing_power = machines[m]["speeds"][v]["processing_power"]
                efficiency = machines[m]["speeds"][v]["efficiency"]
                energy = jobs[i][j]["energy"]
                duration = processing_durations[(i,j,m,v)]

                energy_bias = (processing_power)
                energy_bias += energy / duration / efficiency

                if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                    continue
                powers[t].append((x(i,j,m,t2,v), (energy_bias - idle_power)))

    # Off pre
    for m in machines:
        idle_power = machines[m]["idle_power"]
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t < t2):
                continue
            powers[t].append((turnon(m,t2), -idle_power))

    # Off post
    for m in machines:
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t2 <= t):
                continue

            powers[t].append((turnoff(m,t2), -idle_power))

    # Facility Power
    for t, t2 in product(range(timesteps), repeat=2):
        if not (t2 <= t):
            continue
        powers[t].append((facility(t2), -facility_power))

    for t in range(timesteps):
        if not (t > 0):
            continue
        t_last = t - 1

        powers_c = powers[t]
        powers_l = powers[t_last]

        difference = []
        for variable, bias in powers_c:
            difference.append((variable, bias))
        for variable, bias in powers_l:
            difference.append((variable, -bias))

        for (name1, bias1), (name2, bias2) in product(difference, repeat=2):
            if name1 == name2:
                model.add_linear(v = name1, bias = bias1 * bias2 * weight)
            else:
                model.add_quadratic(u = name1, v = name2, bias = bias1 * bias2 * weight)
    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")


def getViolations(data, sample):
    print("getViolations... ", end="")
    start_time = time.time()

    processing_violations = 0
    overlapping_violations = 0
    sequence_violations = 0
    setup_violations = 0
    maintenance_violations = 0
    machine_off_violations = 0
    facility_violations = 0

    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    processing_durations = data["processing_durations"]
    valid_intervals = data["valid_intervals"]

    for i, j in job_operation_pairs(jobs):
        # jeder Operation muss genau 1 Mal gestartet werden
        variables = [
            (x(i,j,m,t,v), 1)
            for m, v in machine_speed_pairs(machines) for t in range(timesteps)
            if valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]
        ]

        valid = sum([sample.get(variable, 0) * value for variable, value in variables]) == 1
        if not valid:
            processing_violations += 1

        # Operationen dürfen nicht zu spät starten
        for m, v in machine_speed_pairs(machines):
            duration = processing_durations[(i,j,m,v)]
            for t in range(timesteps - duration + 1, timesteps):
                if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                    continue
                valid = sample.get(x(i,j,m,t,v), 0) == 0
                if not valid:
                    processing_violations += 1

    # zu keinem Zeitpuntk darf irgendeine Maschine doppelt belegt sein
    for m in machines:
        for i, i2 in product(jobs, repeat=2):
            for j, j2 in product(jobs[i], jobs[i2]):
                if i == i2 and j == j2:
                    continue

                speeds = machines[m]["speeds"]
                for v, v2 in product(speeds, repeat=2):
                    duration = processing_durations[(i,j,m,v)]
                    for t, t2 in product(range(timesteps), repeat=2):
                        if not (0 <= t2 - t < duration):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                            continue
                        if not (valid_intervals[i2][j2]["start"] <= t2 < valid_intervals[i2][j2]["end"]):
                            continue

                        valid = sample.get(x(i,j,m,t,v), 0) * sample.get(x(i2,j2,m,t2,v2), 0) == 0
                        if not valid:
                            overlapping_violations += 1

    # Operation innerhalb eines Auftrags müssen in der richtigen Reihenfolge bearbeitet werden
    for i in jobs:
        for j, j2 in product(jobs[i], repeat=2):
            for m, v in machine_speed_pairs(machines):
                duration = processing_durations[(i,j,m,v)]
                for t in range(timesteps - duration + 1):
                    for m2, v2 in machine_speed_pairs(machines):
                        for t2 in range(t + duration):
                            if not (j2 > j):
                                continue
                            if not (m2 != m or t2 < t):
                                continue
                            if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                                continue
                            if not (valid_intervals[i][j2]["start"] <= t2 < valid_intervals[i][j2]["end"]):
                                continue

                            valid = sample.get(x(i,j,m,t,v), 0) * sample.get(x(i,j2,m2,t2,v2), 0) == 0
                            if not valid:
                                sequence_violations += 1

    for m in machines:
        setup_duration = machines[m]["setup_duration"]
        speeds = machines[m]["speeds"]

        if setup_duration == 0:
            continue

        # Operationen dürfen nicht zu früh starten
        for t in range(setup_duration):
            for i, j in job_operation_pairs(jobs):
                for v in speeds:
                    if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                        continue
                    valid = sample.get(x(i,j,m,t,v), 0) == 0
                    if not valid:
                        setup_violations += 1


        # Während einer Operation darf keine Setup Zeit einer anderen Operation stattfinden
        for t2 in range(setup_duration, timesteps):
            for (i, j), (i2, j2) in product(job_operation_pairs(jobs), repeat=2):
                for v, v2 in product(speeds, repeat=2):
                    duration = processing_durations[(i,j,m,v)]
                    for t in range(t2):
                        if not (t2 - t < setup_duration + duration):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                            continue
                        if not (valid_intervals[i2][j2]["start"] <= t2 < valid_intervals[i2][j2]["end"]):
                            continue
                        valid = sample.get(x(i,j,m,t,v), 0) * sample.get(x(i2,j2,m,t2,v2), 0) == 0
                        if not valid:
                            setup_violations += 1

    # Wenn ein Auftrag die Maschine wechselt, darf die Setup Zeit nicht die vorangehende Operation zeitlich überlagern
    for i in jobs:
        for j, j2 in product(jobs[i], repeat=2):
            if not (j2 > j):
                continue
            for m, m2 in product(machines, repeat=2):
                speeds = machines[m]["speeds"]
                speeds2 = machines[m2]["speeds"]
                for v, v2 in product(speeds, speeds2):
                    duration = processing_durations[(i,j,m,v)]
                    setup_duration2 = machines[m2]["setup_duration"]
                    for t, t2 in product(range(timesteps), repeat=2):
                        if not (t2 - t < duration + setup_duration2):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                            continue
                        if not (valid_intervals[i][j2]["start"] <= t2 < valid_intervals[i][j2]["end"]):
                            continue
                        valid = sample.get(x(i,j,m,t,v), 0) * sample.get(x(i,j2,m2,t2,v2), 0) == 0
                        if not valid:
                            setup_violations += 1

    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        setup_duration = machines[m]["setup_duration"]
        if maintenance_duration == 0:
            continue

        # Die Wartungsphase muss genau einen Start haben
        variables = [
            (maintenance(m,t), 1)
            for t in range(timesteps)
        ]
        valid = sum([sample.get(variable, 0) * value for variable, value in variables]) == 1
        if not valid:
            maintenance_violations += 1

        # Dieser Start darf nicht zu spät sein
        for t in range(timesteps):
            if not (t > timesteps - maintenance_duration):
                continue
            valid = sample.get(maintenance(m,t), 0) == 0
            if not valid:
                maintenance_violations += 1

        # Wartungsphasen und Operationen dürfen sich nicht überlappen
        speeds = machines[m]["speeds"]
        for t in range(timesteps):
            for i, j in job_operation_pairs(jobs):
                for v in speeds:
                    duration = processing_durations[(i,j,m,v)]

                    # Innerhalb einer Operation/Setupphase darf keine Wartungsphase starten
                    for t2 in range(timesteps):
                        if not (-setup_duration <= t2 - t < duration):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t < valid_intervals[i][j]["end"]):
                            continue
                        valid = sample.get(x(i,j,m,t,v), 0) * sample.get(maintenance(m,t2), 0) == 0
                        if not valid:
                            maintenance_violations += 1

                    # Innerhalb einer Wartungsphase darf keine Operation/Setupphase starten
                    for t2 in range(timesteps):
                        if not (0 <=  t2 - t < maintenance_duration + setup_duration):
                            continue
                        if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                            continue
                        valid = sample.get(maintenance(m,t), 0) * sample.get(x(i,j,m,t2,v), 0) == 0
                        if not valid:
                            maintenance_violations += 1

    # turn on machine
    for m in machines:
        setup_duration = machines[m]["setup_duration"]
        speeds = machines[m]["speeds"]

        # Jede Maschine wird genau einmal eingeschaltet
        variables = [
            (turnon(m,t), 1)
            for t in range(timesteps)
        ]
        valid = sum([sample.get(variable, 0) * value for variable, value in variables]) == 1
        if not valid:
            machine_off_violations += 1

        # Vor dem Einschalten darf keine Wartungsphase starten
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t2 < t):
                continue
            valid = sample.get(turnon(m,t), 0) * sample.get(maintenance(m,t2), 0) == 0
            if not valid:
                machine_off_violations += 1

        # Vor dem Einschalten darf keine Operation/Setupphase starten
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t2 < t + setup_duration):
                continue
            for i, j in job_operation_pairs(jobs):
                for v in speeds:
                    if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                        continue
                    valid = sample.get(turnon(m,t), 0) * sample.get(x(i,j,m,t2,v), 0) == 0
                    if not valid:
                        machine_off_violations += 1

    # turn off machine
    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        speeds = machines[m]["speeds"]

        # Jede Maschine wird genau einmal ausgeschaltet
        variables = [
            (turnoff(m,t), 1)
            # Die Maschine muss bis zum Ende durchlaufen können
            # Da es einen Turnoff-Zeitpunkt geben muss, muss für timesteps+1 eine weitere Variable eingeführt werden
            # Die andere Option wäre, eine Slack-Variable einzuführen, um eine Ungleichungsbedingungs <= 1 zu schaffen
            for t in range(timesteps+1)
        ]
        valid = sum([sample.get(variable, 0) * value for variable, value in variables]) == 1
        if not valid:
            machine_off_violations += 1

        # Die Maschine darf erst nach abgeschlossenen Wartungsphasen ausgeschaltet werden
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t - t2 < maintenance_duration):
                continue
            valid = sample.get(turnoff(m,t), 0) * sample.get(maintenance(m,t2), 0) == 0
            if not valid:
                machine_off_violations += 1

        # Die Maschine darf erst nach abgeschlossenen Operationen ausgeschaltet werden
        for t, t2 in product(range(timesteps), repeat=2):
            for i, j in job_operation_pairs(jobs):
                for v in speeds:
                    duration = processing_durations[(i,j,m,v)]
                    if not (t - t2 < duration):
                        continue
                    if not (valid_intervals[i][j]["start"] <= t2 < valid_intervals[i][j]["end"]):
                        continue
                    valid = sample.get(turnoff(m,t), 0) * sample.get(x(i,j,m,t2,v), 0) == 0
                    if not valid:
                        machine_off_violations += 1

    variables = [
        (facility(t), 1)
        for t in range(timesteps+1)
    ]
    valid = sum([sample.get(variable, 0) * value for variable, value in variables]) == 1
    if not valid:
        facility_violations += 1

    for m in machines:
        for t, t2 in product(range(timesteps+1), repeat=2):
            if not (t2 > t):
                continue
            valid = sample.get(facility(t), 0) * sample.get(turnoff(m,t2), 0) == 0
            if not valid:
                facility_violations += 1

    violations = {
        "processing": processing_violations,
        "overlapping": overlapping_violations,
        "sequence": sequence_violations,
        "setup": setup_violations,
        "maintenance": maintenance_violations,
        "machine_off": machine_off_violations,
        "facility": facility_violations,
    }

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")
    return violations

def getEnergyValue(data, sample):
    print("getEnergyValue... ", end="")
    start_time = time.time()

    total_energy = 0
    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    facility_power = data["facility_power"]
    processing_durations = data["processing_durations"]

    for m in machines:
        idle_power = machines[m]["idle_power"]
        maintenance_power = machines[m]["maintenance_power"]
        maintenance_duration = machines[m]["maintenance_duration"]
        setup_power = machines[m]["setup_power"]
        setup_duration = machines[m]["setup_duration"]
        total_energy += maintenance_duration * (maintenance_power - idle_power)

        speeds = machines[m]["speeds"]
        for i, j in job_operation_pairs(jobs):
            for v in speeds:
                processing_power = speeds[v]["processing_power"]
                efficiency = speeds[v]["efficiency"]
                for t in range(timesteps):
                    if sample.get(x(i,j,m,t,v), 0) == 0:
                        continue
                    duration = processing_durations[(i,j,m,v)]
                    energy = jobs[i][j]["energy"]

                    total_energy += setup_duration * (setup_power - idle_power)
                    total_energy += duration * (processing_power - idle_power)
                    total_energy += energy / efficiency
        total_energy += idle_power * timesteps
        for t in range(timesteps):
            if sample.get(turnon(m,t), 0) == 1:
                total_energy += - t * idle_power
            if sample.get(turnoff(m,t), 0) == 1:
                total_energy += - (timesteps - t) * idle_power

    total_energy += facility_power * timesteps
    for t in range(timesteps):
        if sample.get(facility(t), 0) == 1:
            total_energy += - (timesteps - t) * facility_power

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")
    return total_energy

def getEnergyCostValue(data, sample):
    print("getEnergyCostValue... ", end="")
    start_time = time.time()

    timesteps = data["timesteps"]
    x_norm = [i/timesteps for i in range(timesteps+1)] + [1.0]
    x_price = [2*pi*x for x in x_norm]
    y_price = [0.5*sin(x)+1 for x in x_price]
    # data["price"] = y_price
    # y_price = data["price"]

    total_energy_cost = 0
    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    facility_power = data["facility_power"]
    processing_durations = data["processing_durations"]

    powers = [[] for _ in range(timesteps)]

    # Idle
    for m in machines:
        idle_power = machines[m]["idle_power"]
        total_energy_cost += idle_power * timesteps

    # Maintenance
    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        maintenance_power = machines[m]["maintenance_power"]
        idle_power = machines[m]["idle_power"]
        for t, t2 in product(range(timesteps), repeat=2):
            if not (0 <= t - t2 < maintenance_duration):
                continue
            powers[t].append(sample.get(maintenance(m,t2), 0) * (maintenance_power - idle_power))

    # Setup
    for m, v in machine_speed_pairs(machines):
        setup_duration = machines[m]["setup_duration"]
        setup_power = machines[m]["setup_power"]
        idle_power = machines[m]["idle_power"]
        for i, j in job_operation_pairs(jobs):
            for t, t2 in product(range(timesteps), repeat=2):
                if not (0 < t2 - t <= setup_duration):
                    continue
                powers[t].append(sample.get(x(i,j,m,t2,v), 0) * (setup_power - idle_power))

    # Processing
    for m, v in machine_speed_pairs(machines):
        idle_power = machines[m]["idle_power"]
        for i, j in job_operation_pairs(jobs):
            for t, t2 in product(range(timesteps), repeat=2):
                if not (0 <= t - t2 < processing_durations[(i,j,m,v)]):
                    continue

                processing_power = machines[m]["speeds"][v]["processing_power"]
                efficiency = machines[m]["speeds"][v]["efficiency"]
                energy = jobs[i][j]["energy"]
                duration = processing_durations[(i,j,m,v)]

                energy_bias = (processing_power)
                energy_bias += energy / duration / efficiency

                powers[t].append(sample.get(x(i,j,m,t2,v), 0) * (energy_bias - idle_power))

    # Off pre
    for m in machines:
        idle_power = machines[m]["idle_power"]
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t < t2):
                continue
            powers[t].append(sample.get(turnon(m,t2), 0) * (-idle_power))

    # Off post
    for m in machines:
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t2 <= t):
                continue
            powers[t].append(sample.get(turnoff(m,t2), 0) * (-idle_power))

    # Facility Power
    for t, t2 in product(range(timesteps), repeat=2):
        if not (t2 <= t):
            continue
        powers[t].append(sample.get(facility(t2), 0) * (-facility_power))

    # Price
    for powers, price in zip(powers, y_price):
        total_energy_cost += sum(powers) * price

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

    return total_energy_cost

def getPowerDiffValue(data, sample):
    print("getPowerDiffValue... ", end="")
    start_time = time.time()

    total_power_diff = 0
    jobs = data["jobs"]
    machines = data["machines"]
    timesteps = data["timesteps"]
    facility_power = data["facility_power"]
    processing_durations = data["processing_durations"]

    powers = [[] for _ in range(timesteps)]

    # Maintenance
    for m in machines:
        maintenance_duration = machines[m]["maintenance_duration"]
        maintenance_power = machines[m]["maintenance_power"]
        idle_power = machines[m]["idle_power"]
        for t, t2 in product(range(timesteps), repeat=2):
            if not (0 <= t - t2 < maintenance_duration):
                continue
            powers[t].append(sample.get(maintenance(m,t2), 0) * (maintenance_power - idle_power))

    # Setup
    for m, v in machine_speed_pairs(machines):
        setup_duration = machines[m]["setup_duration"]
        setup_power = machines[m]["setup_power"]
        idle_power = machines[m]["idle_power"]
        for i, j in job_operation_pairs(jobs):
            for t, t2 in product(range(timesteps), repeat=2):
                if not (0 < t2 - t <= setup_duration):
                    continue
                powers[t].append(sample.get(x(i,j,m,t2,v), 0) * (setup_power - idle_power))

    # Processing
    for m, v in machine_speed_pairs(machines):
        idle_power = machines[m]["idle_power"]
        for i, j in job_operation_pairs(jobs):
            for t, t2 in product(range(timesteps), repeat=2):
                if not (0 <= t - t2 < processing_durations[(i,j,m,v)]):
                    continue

                processing_power = machines[m]["speeds"][v]["processing_power"]
                efficiency = machines[m]["speeds"][v]["efficiency"]
                energy = jobs[i][j]["energy"]
                duration = processing_durations[(i,j,m,v)]

                energy_bias = (processing_power)
                energy_bias += energy / duration / efficiency

                powers[t].append(sample.get(x(i,j,m,t2,v), 0) * (energy_bias - idle_power))

    # Off pre
    for m in machines:
        idle_power = machines[m]["idle_power"]
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t < t2):
                continue
            powers[t].append(sample.get(turnon(m,t2), 0) * (-idle_power))

    # Off post
    for m in machines:
        for t, t2 in product(range(timesteps), repeat=2):
            if not (t2 <= t):
                continue
            powers[t].append(sample.get(turnoff(m,t2), 0) * (-idle_power))

    # Facility Power
    for t, t2 in product(range(timesteps), repeat=2):
        if not (t2 <= t):
            continue
        powers[t].append(sample.get(facility(t2), 0) * (-facility_power))

    for t in range(timesteps):
        if not (t > 0):
            continue
        t_last = t - 1

        powers_c = powers[t]
        powers_l = powers[t_last]

        difference = sum(powers_c) - sum(powers_l)
        total_power_diff += difference ** 2

    end_time = time.time()
    print(f"{end_time - start_time:.2f}s")

    return total_power_diff



def useTabu(model: BinaryQuadraticModel, sampleset_size) -> list:
    # sampler = TabuSampler()
    # sampleset = []

    # for i in range(sampleset_size):
    #     energy = -1
    #     sample = None
    #     while energy == -1 or energy >= 1:
    #         sample = sampler.sample(model, num_reads=1, num_sweeps=1000)
    #         energy = sample.first.energy
    #     sampleset.append(sample.samples()[0])

    sampler = TabuSampler()
    sampleset = sampler.sample(model, num_reads=sampleset_size, num_sweeps=1000)
    sampleset = list(sampleset)

    for i in range(len(sampleset)):
        sample = sampleset[i]
        sample = {k: int(v) for k, v in sample.items()}
        sampleset[i] = sample

    return sampleset

def useKerberos(model: BinaryQuadraticModel, sampleset_size, label) -> list:
    from hybrid.reference.kerberos import KerberosSampler
    from dwave.system.samplers import DWaveSampler

    dwave_sampler = DWaveSampler()
    # TODO: aus KerberosSampler().sample() -> hybrid.QPUSubproblemAutoEmbeddingSampler() die ToDos bearbeiten
    sampleset = KerberosSampler().sample(model,
                                        num_reads=sampleset_size,
                                        max_iter=1,
                                        convergence=3,
                                        qpu_reads=100,
                                        qpu_sampler=dwave_sampler,
                                        qpu_params={'label': f"{label} KERBEROS"})

    sampleset = list(sampleset)
    for i in range(len(sampleset)):
        sample = sampleset[i]
        sample = {k: int(v) for k, v in sample.items()}
        sampleset[i] = sample

    return sampleset

def useClassic(model: BinaryQuadraticModel, sampleset_size) -> list:
    from compute.classic_solver import call_bqm_solver_classic

    sampleset = []
    for _ in range(sampleset_size):
        solution, qpu_access_time = call_bqm_solver_classic(model, 3)
        sampleset.append(solution)

    for i in range(len(sampleset)):
        sample = sampleset[i]
        sample = {k: int(v) for k, v in sample.items()}
        sampleset[i] = sample

    return sampleset

def useLeap(model: BinaryQuadraticModel, sampleset_size, label) -> list:

    sampler = LeapHybridBQMSampler()
    sampleset = sampler.sample(model, label=f"{label} LEAP")

    print(json.dumps(sampleset.info, indent=4))

    sampleset = list(sampleset)
    for i in range(len(sampleset)):
        sample = sampleset[i]
        sample = {k: int(v) for k, v in sample.items()}
        sampleset[i] = sample

    return sampleset
