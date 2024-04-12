from typing import List
import pandas as pd
import os
from dataclasses import dataclass, field

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

@dataclass
class CloudInstanceInfo:
    instance_type: str
    cloud: str
    region: str
    instance_num: int

    def copy(self):
        return CloudInstanceInfo(
            instance_type=self.instance_type,
            cloud=self.cloud,
            region=self.region,
            instance_num=self.instance_num,
        )

# Mapping from GPU type to cloud instance info
A10G_INSTANCE = CloudInstanceInfo(
    instance_type="g5.xlarge",
    cloud="aws",
    region="us-west-2",
    instance_num=1,
)
A100_80GB_INSTANCE = CloudInstanceInfo(
    instance_type="Standard_NC24ads_A100_v4",
    cloud="azure",
    region="eastus",
    instance_num=1,
)

GPU_INSTANCE_MAP = {
    "A10G": A10G_INSTANCE,
    "A100-80GB": A100_80GB_INSTANCE,
}

def get_instance_info(gpu_type: str, num_of_instance: int) -> CloudInstanceInfo:
    instance_info = GPU_INSTANCE_MAP[gpu_type].copy()
    instance_info.instance_num = num_of_instance

    return instance_info