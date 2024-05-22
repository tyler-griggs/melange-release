# GPU Profiling

## About
This directory holds the code we use to profile GPU performance to find their throughputs and latencies. The bash script `benchmark-launcher.sh` is used to launch multiple sequential instances of `gpu-benchmark.py`. Each instance of `gpu-benchmark.py` profiles a specific request size and rate. 

## Launching Benchmarks
First, deploy your model of choice on the GPU you wish to profile. We use [vLLM](https://github.com/vllm-project/vllm/tree/main) as our inference engine, which can be launched by following the instructions in their github repo. Once your model is up and running, modify `benchmark-launcher.sh` to configure which request sizes and rates should be benchmarked. Finally, simply run `bash benchmark-launcher.sh` and, upon script completion, the resuts will be in the configured result directory.