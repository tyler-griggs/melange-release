from typing import List
import pandas as pd
import os


# Convert max throughput profiling to a mapping from request size to load
def tputs_to_loads_2d(max_tputs: List[List[float]]):
    loads = []
    for i in range(len(max_tputs)):
        loads.append([])
        for j in range(len(max_tputs[0])):
            load = 1 / max_tputs[i][j]
            loads[-1].append(load)
    return loads


def display_experiment_results(results, solver_labels, ilp_result):
    df = pd.DataFrame(results)
    df.fillna(0, inplace=True)

    # Add the last column filled with zeros
    df["Savings"] = [
        str(round(((x["cost"] - ilp_result["cost"]) / x["cost"]) * 100, 2)) + "%"
        for x in results
    ]

    # Ensure the 'cost' column is second to last and 'LastColumn' is last, this step might need adjustment based on actual GPU types
    # Assuming we don't know all GPU types in advance, let's dynamically sort columns except for 'cost' and 'LastColumn'
    gpu_columns = [col for col in df.columns if col not in ["cost", "Savings"]]
    sorted_columns = gpu_columns + ["cost", "Savings"]
    df = df[sorted_columns]
    df.index = solver_labels

    # Display the table
    print(df)


# Set up your huggingface access token
def set_hf_token(access_token):
    os.environ["HF_TOKEN"] = access_token
