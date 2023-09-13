import subprocess
import time
import os
import json

def count(path_list, model_solver_name, check_valid=False):

    occurences = 0
    for path in path_list:
        if not os.path.isfile(path):
            continue
        if model_solver_name not in path:
            continue

        if check_valid:
            with open(path, 'r') as f:
                data = json.load(f)
                violations = data["stats"]["violations"]
                total_violations = sum(violations.values())
                if total_violations == 0:
                    occurences += 1
        else:
            occurences += 1

    return occurences

def fetch_queue():
    result_squeue = subprocess.Popen("squeue", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output_squeue, error_squeue = result_squeue.communicate()

    in_queue = {}

    lines = output_squeue.split(b'\n')
    lines = [line.decode('utf-8') for line in lines]
    lines = [line for line in lines if "JOBID" not in line and line != '']
    for i, line in enumerate(lines):
        while "  " in line:
            line = line.replace("  ", " ")
        line = line.split(' ')
        lines[i] = line
        s = line[3]
        in_queue[s] = in_queue.get(s, 0) + 1

    return in_queue


jobs = {
    # 0: {
    #     "model": "bqm_40x8(6)x50(35)_e",
    #     "short": "k40e",
    #     "scalings": [10*(i+1) for i in range(10)],
    #     "samples": 10,
    #     "check_valid": False,
    #     "solver": "bqm-kerberos",
    #     "memory": 10,
    #     "time_limit": "00:15:00"
    # },
    1: {
        "model": "bqm_40x8(6)x50(35)_p",
        "short": "c40p",
        "scalings": range(350, 1001, 50),
        "samples": 10,
        "check_valid": False,
        "solver": "bqm-classic",
        "memory": 13,
        "time_limit": "00:15:00"
    },
    # 2: {
    #     "model": "bqm_40x8(6)x50(35)_p",
    #     "short": "k40p",
    #     "scalings": [10*(i+1) for i in range(30)],
    #     "samples": 10,
    #     "check_valid": False,
    #     "solver": "bqm-kerberos",
    #     "memory": 16,
    #     "time_limit": "00:15:00"
    # },
}

directory_path = "./output/"
total_samples = sum([jobs[job]["samples"] * len(jobs[job]["scalings"]) for job in jobs])

max_in_parallel = 25
done = False
while not done:
    queue = fetch_queue()
    queue_len = sum(queue.values())

    job_starts = 0
    total_running = 0
    total_completed = 0
    for job in jobs:
        model = jobs[job]["model"]
        short = jobs[job]["short"]
        scalings = jobs[job]["scalings"]
        samples = jobs[job]["samples"]
        check_valid = jobs[job]["check_valid"]
        solver = jobs[job]["solver"]
        memory = jobs[job]["memory"]
        time_limit = jobs[job]["time_limit"]

        for scaling in scalings:
            model_name = f"{model}{scaling}"
            job_name = f"{short}{scaling}"

            file_list = os.listdir(directory_path)
            path_list = [directory_path + file_name for file_name in file_list]
            completed = count(path_list, f"{model_name}_{solver}", check_valid)
            completed = min(completed, samples)
            total_completed += completed
            running = queue.get(str(job_name), 0)
            total_running += running
            remaining = samples - completed - running
            job_starts += remaining

            for r in range(remaining):
                if queue_len >= max_in_parallel:
                    break
                command = f'sbatch --mem={memory}G --time={time_limit} --job-name={job_name} sample "models/{model}{scaling}.pickle" {solver}'

                result_sbatch = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                output_batch, error_batch = result_sbatch.communicate()
                queue_len += 1

    progress = f"{total_completed/total_samples*100:.2f}%"
    print(f"Progress: {progress} ({total_completed}/{total_samples})")

    if total_completed >= total_samples:
        done = True
        break

    time.sleep(60)