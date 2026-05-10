import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import json
import glob


def visualize_partition_timings():

    # ---------------------------------------------------------
    # 1. Load and Parse the JSON Data
    # ---------------------------------------------------------
    # Find all JSON files matching your naming convention
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(cur_dir, "results")
    file_pattern = "single_layer_timing_data_*.json"
    file_pattern = os.path.join(results_dir, file_pattern)
    files = glob.glob(file_pattern)

    print(f"Found {len(files)} timing data files: {files}")

    data_records = []

    for filepath in files:
        with open(filepath, "r") as file:
            data = json.load(file)

            # Iterate through the dictionary keys
            for key, val in data.items():
                # Skip the meta_data block
                if key == "meta_data":
                    continue

                # Extract the relevant fields
                operation = val.get("operation")
                system_size = val.get("system_size")
                world_size = val.get("world_size")

                # We append EVERY elapsed time as its own row.
                # This allows Seaborn to calculate variance/error bands automatically.
                times = val.get("elapsed_times_ms", [])
                if len(val.get("elapsed_times_ms", [])) > 20:
                    times = times[10:]  # Skip the first 10 iterations as warmup

                for time_ms in times:
                    data_records.append(
                        {
                            "Operation": operation,
                            "System Size": system_size,
                            "World Size": world_size,
                            "Time (ms)": time_ms,
                        }
                    )

    # Convert to a pandas DataFrame for easy plotting
    df = pd.DataFrame(data_records)
    print(df)

    # ---------------------------------------------------------
    # 2. Generate Plots (One per Operation)
    # ---------------------------------------------------------
    # Set a clean visual theme
    sns.set_theme(style="whitegrid")

    # Get a list of all unique operations found in the data
    operations = df["Operation"].dropna().unique()

    for op in operations:
        # Filter the dataframe for the current operation
        op_df = df[df["Operation"] == op]

        # Create a new figure for each operation
        plt.figure(figsize=(10, 6))

        # Plot the timing curves
        # Seaborn automatically aggregates the multiple iterations into a mean line
        # and adds a shaded confidence interval around it.
        ax = sns.lineplot(
            data=op_df,
            x="System Size",
            y="Time (ms)",
            hue="World Size",
            palette="tab10",  # Color palette for different world sizes
            marker="o",  # Puts a dot at each data point
            err_style="band",  # Shows the shaded variance (default is 95% CI)
        )

        # ---------------------------------------------------------
        # 3. Formatting the Axes (Crucial for System Sizes)
        # ---------------------------------------------------------
        # Because your system sizes jump from 100 to 10,000,000,
        # a log-log scale is required to actually see the trends.
        plt.xscale("log")
        plt.yscale("log")

        # Titles and Labels
        plt.title(
            f"Timing Trends: {op.replace('_', ' ').title()}", weight="bold", fontsize=14
        )
        plt.xlabel("System Size (Log Scale)", fontsize=12)
        plt.ylabel("Execution Time in ms (Log Scale)", fontsize=12)

        # Clean up the legend
        plt.legend(
            title="World Size (GPUs)", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.tight_layout()

        # Display the plot (or you can use plt.savefig(f"{op}_timing.png") to save them)
        # plt.show()
        figures_dir = os.path.join(cur_dir, "results", "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        plt.savefig(f"results/figures/{op}_timing.png")
        print(f"Plot saved for operation: {op}")


if __name__ == "__main__":

    visualize_partition_timings()
