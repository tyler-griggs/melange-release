# Mélange: Cost Efficient Large Language Model Serving by Exploiting GPU Heterogeneity

## About
This repository provides the implementation of Mélange used in our paper. We open source this tool with the hope that it can be useful for researchers and practitioners to experiment with utilizing heterogeneous GPUs to reduce cloud instance costs for LLM serving.

## Getting Started
```bash
# Tested on Python 3.9.18

# 1. Install the necessary dependencies
pip install -r requirements.txt

# 2. Execute the solver with a specific input configuration
python -m melange.main -c melange/config/example.json

# 3. By default, the solver will save the output in a JSON file named as "melange_result.json" at the root directory
```


## Explanation of Inputs and Outputs
### Inputs
The solver requires a json file with the following inputs:
1. `workload_distribution`: A 2D matrix representing the distribution of input and output lengths that the LLM service expects. Each row refers to one input size, each column refer to one output size, and each cell correspond to the proportion of requests that are within the cell's input and output size range (i.e., a bucket). The request size boundaries between buckets can be tuned to reach a desired balance of granularity and solver complexity. An example for the range of input and output sizes could be as follows:
    - Input/Output size: 1-25, 25-100, 100-250, ...
    - The cell at (0, 0) represents the request rate for requests with input and output sizes of 1-25 tokens.
    - The cell at (0, 1) represents the request rate for requests with input size 1-25 tokens and output size 25-100 tokens.
    - And so on ...
2. `gpu_info`: A list of dictionaries, where each dictionary contains the following keys:
    - `name`: The name of the GPU.
    - `cost`: The hourly rental cost of the GPU.
    - `tputs`: A 2D matrix where each cell represents the GPU's profiled maximum throughput for requests of size equivalent to the corresponding cell in the `workload_distribution` matrix.
3. `total_request_rate`: A float value representing the total request rate of the workload.
4. `slice_factor`: An integer multiplier for the number of slices each bucket is split into.

Please kindly refer to [example.json](melange/config/example.json) for an example of the inputs and check out our paper for more details on our methodology.

### Outputs
### Solver Output
The solver returns a dictionary containing the following:
1. The name of each GPU and the number of that GPU type to use.
2. The total cost for one hour.

An example of the solver output is as follows:
```json
{
    "A10G": 10,
    "A100-80GB": 0,
    "cost": 10.1
}
```
In this case, the solver recommends using 10 A10G GPUs and 0 A100-80GB GPUs, which results in a total cost of $10.10/hr.

### Output Formats
Melange currently supports the following output formats:
* **JSON**:
  * Default output format.
  * The solver output is saved as a JSON file at the root directory with the name `melange_result.json`.
* [**SkyPilot**](https://github.com/skypilot-org/skypilot/tree/master)-compatible configuration file:
  * Each GPU instance is saved as a separate configuration file.
  * The generated yaml files are based on an [existing template](melange/output/skypilot_template.yaml) and can be directly used with SkyPilot.
  * See the [example usage](#example-usage-to-integrate-with-skypilot) for more details.

Supported GPU types and information when exporting so far:
  * {Cloud}: {GPU_type}, {instance_type}, {region}, {$/hr}
  * AWS: A10G, g5.xlarge, us-west-2, 1.01
  * Azure: A100-80, Standard_NC24ads_A100_v4, eastus, 3.67

Support of more output formats and GPU types together with user-provided output templates are planned for future releases.

## Run with Your Own Dataset or GPU Information
The toy example at [script_code](melange/main.py) and [example_config](melange/config/example.json) includes examples of the four inputs to Mélange, which should be replaced to fit your setting's need.

### Workload Distribution
   1. Determine the expected distribution of request sizes your LLM service expects. For example, you can use historical data of requests served by your service. In our evaluations, we used publicly available datasets (such as [Chatbot Arena](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)) to determine a reasonable distribution of request sizes.
   2. Populate the `workload_distribution` based on the determined distribution. As mentioned, each row refers to a single input size, each column refers to a single output size, and each cell corresponds to the proportion of requests that fall into the given bucket. For example, a cell value of 0.1 indicates that 10% are in that bucket's size range.

### GPU Information
For each GPU instance of interest, provide the following information:
   1. The name of the instance.
   2. The hourly rental cost of the instance.
   3. Results from profiling the GPUs maximum throughput (in requests/s) for requests within each bucket's size range from the buckets in `workload_distribution`.

### Overall Rate and Slice Factor
 1. Determine the service's overall request rate across all request sizes, and provide it as the `total_request_rate`.
 2. Decide on the slice factor. We find that the solver's output is not very sensitive to the choice of slice factor. We empirically find that 4 is sufficient for most cases.

## Example usage to integrate with SkyPilot
In this example, we will deploy one AWS `g5.xlarge` instance in `us-west-2`. This instance will be serving Llama-2 with vLLM's OpenAI-compatible API server. This example is adapted from [SkyPilot](https://github.com/skypilot-org/skypilot/blob/master/llm/vllm/README.md) and you can find more details on the [template SkyPilot configuration file](melange/output/skypilot_template.yaml).

### Prerequisites
1. Install the dependencies for SkyPilot by following the instructions at [SkyPilot](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html).
2. Please [set up your AWS credentials]((https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#amazon-web-services-aws)), and ensure you have sufficient compute quota for deployment.
3. Please ensure you have [the Llama-2 model access and your HuggingFace access token](https://github.com/skypilot-org/skypilot/blob/master/llm/llama-2/README.md#pre-requisites).

### Walkthrough
* Install the necessary dependencies.
```bash
pip install -r requirements.txt
```

* Execute the solver with a specific input configuration.
```bash
python -m melange.main -c melange/config/example.json -f skypilot
```

* The output will be saved as a SkyPilot-compatible configuration file named as `skypilot_A10G_0.yaml` in the root directory.

* Now, start serving the model using SkyPilot.
```bash
sky launch -c melange-skypilot skypilot_A10G_0.yaml --env HF_TOKEN=YOUR_HUGGING_FACE_API_TOKEN
```

* Wait until the instance is set up. Upon completion, you should see the output like these.
```bash
(task, pid=29834) INFO:     Started server process [30072]
(task, pid=29834) INFO:     Waiting for application startup.
(task, pid=29834) INFO:     Application startup complete.
(task, pid=29834) INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

* Check the IP address of the instance and make a query to the model.
```bash
IP=$(sky status --ip vllm-llama2)
curl http://$IP:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "meta-llama/Llama-2-7b-chat-hf",
      "prompt": "San Francisco is a",
      "max_tokens": 7,
      "temperature": 0
  }'
```

*  A similar output should be received.
```bash
{
    "id":"cmpl-70ed1fb2f660492bb0a476c8a19cab7d",
    "object":"text_completion","created":706,
    "model":"meta-llama/Llama-2-7b-chat-hf",
    "choices":[{
        "index":0,
        "text":"city in Northern California that is known",
        "logprobs":null,
        "finish_reason":"length"
    }],
    "usage":{
        "prompt_tokens":5,
        "total_tokens":12,
        "completion_tokens":7
    }
}
```

*  Terminate the cluster when you are done. Note that all instances will be removed and cannot be recovered.
```bash
sky down melange-skypilot
```

## For Arm-based Mac platforms
We have occasionally (but not always) seen errors using PuLP on Arm-based MACs (M1/M2/M3). If you experience this issue, it's likely because the default ILP solver used by the PuLP library is not compatible with your architecture and will require additional steps.
1. Install the COIN CBC ILP solver using homebrew: `brew install coin-or-tools/coinor/cbc`
2. In [melange/solver.py](melange/solver.py), uncomment the following code to use the CBC solver. Note that your `path` may differ based on where the library was installed.
```
solver= pulp.getSolver('COIN_CMD', path='/opt/homebrew/opt/cbc/bin/cbc', msg=0)
problem.solve(solver)
```
