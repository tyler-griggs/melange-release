# Mélange: Cost Efficient Large Language Model Serving by Exploiting GPU Heterogeneity

## About
This repository provides the implementation of Mélange used in our paper. We open source this tool with the hope that it can be useful for researchers and practitioners to experiment with utilizing heterogeneous GPUs to reduce cloud instance costs for LLM serving.

## Getting Started
```bash
# Tested on Python 3.9.18

# 1. Install the necessary dependencies
pip install -r requirements.txt

# 2. Execute the solver
python -m melange.main
```

## Explanation of Inputs and Outputs
### Inputs
The solver requires the following inputs:
1. `workload_distribution`: A 2D matrix representing the distribution of input and output lengths that the LLM service expects. Each row refers to one input size, each column refer to one output size, and each cell correspond to the proportion of requests that are within the cell's input and output size range (i.e., a bucket). The request size boundaries between buckets can be tuned to reach a desired balance of granularity and solver complexity. An example for the range of input and output sizes could be as follows:
    - Input/Output size: 1-25, 25-100, 100-250, ...
    - The cell at (0, 0) represents the request rate for requests with input and output sizes of 1-25 tokens.
    - The cell at (0, 1) represents the request rate for requests with input size 1-25 tokens and output size 25-100 tokens.
    - And so on ...
2. `gpu_info`: A list of dictionaries, where each dictionary contains the following keys:
    - 'name': The name of the GPU.
    - 'cost': The hourly rental cost of the GPU.
    - 'tputs': A 2D matrix where each cell represents the GPU's profiled maximum throughput for requests of size equivalent to the corresponding cell in the `workload_distribution` matrix.
3. `overall_rate`: A float value representing the total request rate of the workload.
4. `slice_factor`: An integer multiplier for the number of slices each bucket is split into.

Please kindly refer to [script_code](melange/main.py) for an example of the inputs and check out our paper for more details on our methodology.

### Outputs
The solver returns a dictionary containing the following:
1. The name of each GPU and the number of that GPU type to use.
2. The total cost for one hour.

An example of the output is as follows:
```json
{
    "A10G": 10.0,
    "A100": 0.0,
    "cost": 10.1
}
```
In this case, the solver recommends using 10 A10G GPUs and 0 A100 GPUs, which results in a total cost of $10.10/hr.


## Run with Your Own Dataset or GPU Information
The toy example at [script_code](melange/main.py) includes examples of the four inputs to Mélange, which should be replaced to fit your setting's need.

### Workload Distribution
   1. Determine the expected distribution of request sizes your LLM service expects. For example, you can use historical data of requests served by your service. In our evaluations, we used publicly available datasets (such as [Chatbot Arena](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)) to determine a reasonable distribution of request sizes. 
   2. Populate the `workload_distribution` based on the determined distribution. As mentioned, each row refers to a single input size, each column refers to a single output size, and each cell corresponds to the proportion of requests that fall into the given bucket. For example, a cell value of 0.1 indicates that 10% are in that bucket's size range.

### GPU Information
For each GPU instance of interest, provide the following information:
   1. The name of the instance.
   2. The hourly rental cost of the instance.
   3. Results from profiling the GPUs maximum throughput (in requests/s) for requests within each bucket's size range from the buckets in `workload_distribution`.

### Overall Rate and Slice Factor
 1. Determine the service's overall request rate across all request sizes, and provide it as the `overall_rate`.
 2. Decide on the slice factor. We find that the solver's output is not very sensitive to the choice of slice factor. We empirically find that 4 is sufficient for most cases.

## Repo To-Do List
- [ ] Release more scripts or tools to facilitate the process of analyzing the dataset and/or profiling the GPUs.
- [ ] Maintain a list of GPU profiles for popular GPUs to make it easier for users to use the solver.
