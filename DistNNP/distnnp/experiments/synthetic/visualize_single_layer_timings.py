import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np


def visualize_partition_timings(timing_data_json_path):
    with open(timing_data_json_path, "r") as f:
        data = json.load(f)

    keys = data.keys()

    ops = set([x.split(" - ")[0] for x in list(keys)])

    for op in ops:
        system_sizes = []
        parsed_data = []
        for key, metrics in data.items():
            # Extract system size from the string key (e.g. "Partition - system size 1000" -> 1000)
            sys_size = int(key.split("size ")[-1])
            system_sizes.append(sys_size)
            op_name = key.split(" - ")[0]

            if op_name != op:
                continue

            # Exclude the first run because it's a CUDA warmup iteration
            times = metrics["elapsed_times_ms"][5:]
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(
                f"Processing key: {key}, extracted op: {op_name}, system size: {sys_size} atoms, ",
                f"average time (excluding warmup): {avg_time:.2f} ms ± {std_time:.2f} ms (n={len(times)})",
            )

            for t in times:
                parsed_data.append(
                    {
                        "System Size": sys_size,
                        "Execution Times (ms)": t,
                    }
                )

        # Convert to DataFrame and sort by system size so the bars are in order
        df = pd.DataFrame(parsed_data)
        df = df.sort_values(by="System Size")

        # 2. Setup the plot
        plt.figure(figsize=(9, 6))
        sns.set_theme(style="whitegrid")  # Nice clean background

        # 3. Create the bar plot
        sns.lineplot(
            data=df,
            x="System Size",
            y="Execution Times (ms)",
            marker="o",  # Add a dot at each data point
            markersize=12,
            errorbar="sd",  # Automatically calculate and shade standard deviation
            color="blue",
        )

        # 4. Enforce continuous proportional scaling
        plt.xlim(
            -1000, max(system_sizes) * 1.1
        )  # Force the graph to show the scale from zero
        plt.xticks(
            system_sizes, rotation=90, fontsize=10
        )  # Set x-ticks to be the system sizes

        # 5. Formatting
        if op == "Build Halo Maps":
            plt.yscale(
                "log"
            )  # Use logarithmic scale for better visibility of differences

        plt.title(f"{op} Time by System Size", fontsize=14, pad=15)
        plt.ylabel("Average Time (ms)", fontsize=12)
        plt.xlabel(
            "System Size (# of atoms)",
            fontsize=12,
        )
        plt.tight_layout()

        figure_dir = os.path.dirname(timing_data_json_path)
        figure_dir = os.path.join(figure_dir, "figures")
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        plt.savefig(os.path.join(figure_dir, f"cuda_timings_{op}.png"))
        plt.close()


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    timing_data_json_path = os.path.join(cur_dir, "results", "timing_data.json")
    visualize_partition_timings(timing_data_json_path)
