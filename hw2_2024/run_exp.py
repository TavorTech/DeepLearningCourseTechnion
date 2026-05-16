#!/usr/bin/env python3

import os
import argparse
import time
import psutil
from threading import Thread, Event
from hw2.experiments import cnn_experiment
import subprocess

# Base configuration for the experiment
OUT_DIR = "./results"

# Experiment configurations
EXPERIMENT_1_1_DEPTHS = [2, 4, 8, 16]
EXPERIMENT_1_1_FILTERS = [64, 32]  # Ensures 64 is executed first
EXPERIMENT_1_2_DEPTHS = [2, 4, 8]
EXPERIMENT_1_2_FILTERS = [32, 64, 128]
EXPERIMENT_1_3_DEPTHS = [2, 3, 4]
EXPERIMENT_1_3_FILTERS = [[64, 128]]  # Fixed K=[64, 128] for exp1_3
EXPERIMENT_1_4_DEPTHS = [8, 16, 32]  # New configuration for exp1_4
EXPERIMENT_1_4_FILTERS = [[32], [64, 128, 256]]  # Fixed K=[32] and K=[64, 128, 256] for exp1_4

# Ensure the output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

def monitor_system_usage(stop_event, interval=10):
    """
    Monitors system usage periodically to display CPU and memory utilization.

    :param stop_event: Event to signal when monitoring should stop.
    :param interval: Time in seconds between usage checks.
    """
    try:
        while not stop_event.is_set():
            print("\n*** System Usage Statistics ***")
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            print(f"CPU Usage: {cpu_usage}%")
            print(f"Memory Usage: {memory.used / 1024 ** 2:.2f} MB / {memory.total / 1024 ** 2:.2f} MB")
            time.sleep(interval)
    except Exception as error:
        print(f"Failed to fetch system usage statistics: {error}")

def execute_local(run_name, filters, depth):
    """
    Executes the experiment locally with system usage monitoring using cnn_experiment directly.

    :param run_name: Unique name for the experiment run
    :param filters: List of filters per layer (K)
    :param depth: Number of layers per block (L)
    """
    print(f"Executing experiment locally: {run_name}")

    # Start system monitoring in a separate thread with a stop event
    stop_event = Event()
    monitor_thread = Thread(target=monitor_system_usage, args=(stop_event, 5), daemon=True)
    monitor_thread.start()

    try:
        # Run the experiment
        cnn_experiment(
            run_name,
            out_dir=OUT_DIR,
            seed=None,
            bs_train=128,
            batches=100,
            epochs=10,
            early_stopping=3,
            filters_per_layer=filters,
            layers_per_block=depth,
            pool_every=2,
            hidden_dims=[100],
            model_type="cnn",
        )
    finally:
        # Signal the monitoring thread to stop
        stop_event.set()
        monitor_thread.join()

def execute_srun(run_name, filters, depth):
    """
    Executes the experiment on a server using srun.

    :param run_name: Unique name for the experiment run
    :param filters: List of filters per layer (K)
    :param depth: Number of layers per block (L)
    """
    cmd = [
        "srun", "-c", "2", "--gres=gpu:1", "--pty",
        f"--job-name={run_name}", "python3", "-m", "hw2.experiments", "run-exp",
        "-n", run_name,
        "-o", OUT_DIR,
        "-K", *map(str, filters),
        "-L", str(depth),
        "-P", "2",
        "-H", "100"
    ]
    print(f"Command to execute on server: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_experiment(run_name, filters, depth, local=False):
    """
    Runs an experiment and ensures the results are saved to a JSON file.

    :param run_name: Name of the run (for result identification)
    :param filters: List of filters per layer (K)
    :param depth: Number of layers per block (L)
    :param local: Whether to run the experiment locally.
    """
    print(f"Starting experiment: {run_name} with K={filters}, L={depth}")
    if local:
        execute_local(run_name, filters, depth)
    else:
        execute_srun(run_name, filters, depth)

def run_experiment_1_1(local):
    """
    Runs Experiment 1.1: Varying the network depth (L) with fixed filters (K).
    """
    for filters in EXPERIMENT_1_1_FILTERS:
        for depth in EXPERIMENT_1_1_DEPTHS:
            run_name = f"exp1_1_L{depth}_K{filters}"
            run_experiment(run_name, [filters], depth, local=local)

def run_experiment_1_2(local):
    """
    Runs Experiment 1.2: Varying the number of filters per layer (K) with fixed depth (L).
    """
    for depth in EXPERIMENT_1_2_DEPTHS:
        for filters in EXPERIMENT_1_2_FILTERS:
            run_name = f"exp1_2_L{depth}_K{filters}"
            run_experiment(run_name, [filters], depth, local=local)

def run_experiment_1_3(local):
    """
    Runs Experiment 1.3: Varying both the number of filters (K) and network depth (L).
    Each (K, L) combination generates a separate run.
    """
    for depth in EXPERIMENT_1_3_DEPTHS:
        for filters in EXPERIMENT_1_3_FILTERS:
            run_name = f"exp1_3_L{depth}_K{filters[0]}-{filters[1]}"
            run_experiment(run_name, filters, depth, local=local)

def run_experiment_1_4(local):
    """
    Runs Experiment 1.4: Varying the network depth (L) with fixed filters (K).
    """
    for filters in EXPERIMENT_1_4_FILTERS:
        for depth in EXPERIMENT_1_4_DEPTHS:
            run_name = f"exp1_4_L{depth}_K{'-'.join(map(str, filters))}"
            run_experiment(run_name, filters, depth, local=local)

def print_help():
    """
    Prints the help message with details of all available experiments.
    """
    print("Available Experiments:")
    print("  exp1_1: Vary network depth (L) with K=32 and K=64")
    print("  exp1_2: Vary number of filters per layer (K) with L=2, L=4, L=8")
    print("  exp1_3: Vary both filters (K=[64,128]) and network depth (L=2,3,4)")
    print("  exp1_4: Vary network depth (L) with K=[32] and K=[64, 128, 256]")
    print("Usage Examples:")
    print("  python3 run_exp.py --experiment exp1_1")
    print("  python3 run_exp.py --experiment exp1_1 --local")
    print("  python3 run_exp.py --experiment exp1_2 --local --K 64 --L 4")
    print("  python3 run_exp.py --experiment exp1_3")
    print("  python3 run_exp.py --experiment exp1_4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("--experiment", type=str, help="Specify the experiment to run.", default="exp1_1")
    parser.add_argument("--local", action="store_true", help="Run locally without srun.")
    parser.add_argument("--K", type=int, help="Specify the number of filters per layer (K).", default=None)
    parser.add_argument("--L", type=int, help="Specify the number of layers per block (L).", default=None)
    parser.add_argument("--help-flag", action="store_true", help="Show available experiments and usage.")

    args = parser.parse_args()

    if args.help_flag:
        print_help()
    elif args.experiment == "exp1_1":
        if args.K is not None and args.L is not None:
            run_name = f"exp1_1_L{args.L}_K{args.K}"
            run_experiment(run_name, [args.K], args.L, local=args.local)
        else:
            print("Running Experiment 1.1.")
            run_experiment_1_1(local=args.local)
    elif args.experiment == "exp1_2":
        if args.K is not None and args.L is not None:
            run_name = f"exp1_2_L{args.L}_K{args.K}"
            run_experiment(run_name, [args.K], args.L, local=args.local)
        else:
            print("Running Experiment 1.2.")
            run_experiment_1_2(local=args.local)
    elif args.experiment == "exp1_3":
        if args.K is not None and args.L is not None:
            run_name = f"exp1_3_L{args.L}_K{args.K}"
            run_experiment(run_name, [args.K], args.L, local=args.local)
        else:
            print("Running Experiment 1.3.")
            run_experiment_1_3(local=args.local)
    elif args.experiment == "exp1_4":
        if args.K is not None and args.L is not None:
            run_name = f"exp1_4_L{args.L}_K{args.K}"
            run_experiment(run_name, [args.K], args.L, local=args.local)
        else:
            print("Running Experiment 1.4.")
            run_experiment_1_4(local=args.local)
    else:
        print("Invalid experiment specified. Use --help-flag to see available experiments.")