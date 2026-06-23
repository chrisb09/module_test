#!/usr/bin/env python3
import os
import glob
import csv
import sys
import statistics

def main():
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = "timings/phydll_run_rank_*.csv"
    
    files = glob.glob(pattern)
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} timing files.")

    # Data structures to collect timings
    # Keys will be step index
    step_durations = {} # step -> list of durations in us
    all_ranks_data = [] # list of dicts for global metrics

    min_start_time = float('inf')
    max_end_time = float('-inf')

    for filepath in files:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rank = int(row['Rank'])
                    step = int(row['Step'])
                    start_ns = int(row['StartTime_ns'])
                    end_ns = int(row['EndTime_ns'])
                    duration_us = int(row['Duration_us'])

                    min_start_time = min(min_start_time, start_ns)
                    max_end_time = max(max_end_time, end_ns)

                    if step not in step_durations:
                        step_durations[step] = []
                    step_durations[step].append(duration_us)

                    all_ranks_data.append({
                        'rank': rank,
                        'step': step,
                        'start_ns': start_ns,
                        'end_ns': end_ns,
                        'duration_us': duration_us
                    })
                except (ValueError, KeyError) as e:
                    print(f"Error parsing row in {filepath}: {e}")

    if not all_ranks_data:
        print("No valid timing data parsed.")
        return

    total_elapsed_sec = (max_end_time - min_start_time) / 1e9
    print("\n=== Global Run Metrics ===")
    print(f"Total time span across all ranks: {total_elapsed_sec:.4f} seconds")
    print(f"Number of ranks reporting: {len(files)}")

    print("\n=== Step-by-Step Inference Durations (Across All Ranks) ===")
    print(f"{'Step':<6} | {'Min (s)':<10} | {'Max (s)':<10} | {'Mean (s)':<10} | {'Median (s)':<10} | {'StdDev (s)':<10}")
    print("-" * 72)
    
    for step in sorted(step_durations.keys()):
        durations_s = [d / 1e6 for d in step_durations[step]]
        d_min = min(durations_s)
        d_max = max(durations_s)
        d_mean = statistics.mean(durations_s)
        d_median = statistics.median(durations_s)
        d_std = statistics.stdev(durations_s) if len(durations_s) > 1 else 0.0
        print(f"{step:<6} | {d_min:<10.4f} | {d_max:<10.4f} | {d_mean:<10.4f} | {d_median:<10.4f} | {d_std:<10.4f}")

    # Steady state analysis (excluding step 0)
    steady_state_durations = []
    for step, durations in step_durations.items():
        if step > 0:
            steady_state_durations.extend(durations)

    print("\n=== Summary Stats ===")
    if 0 in step_durations:
        step_0_s = [d / 1e6 for d in step_durations[0]]
        print(f"Step 0 (Startup/Warmup) Mean Duration: {statistics.mean(step_0_s):.4f} s (Min: {min(step_0_s):.4f}s, Max: {max(step_0_s):.4f}s)")
    
    if steady_state_durations:
        steady_s = [d / 1e6 for d in steady_state_durations]
        print(f"Steady State (Steps 1+) Mean Duration: {statistics.mean(steady_s):.4f} s (Min: {min(steady_s):.4f}s, Max: {max(steady_s):.4f}s, StdDev: {statistics.stdev(steady_s):.4f}s)")
    else:
        print("No steady state steps recorded (only step 0).")

if __name__ == "__main__":
    main()
