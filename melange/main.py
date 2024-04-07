from scripts.solver import HeteroAcceleratorSolver


def solver_run_example():
    #### Input required for the solver ####
    #### Replace the following with your own input ####

    # Each "tputs" is a 2D matrix based on offline profiling.
    # Each cell inside represents the GPU's profiled maximum throughput.
    # It corresponds to each of the request sizes in the `workload_distribution` matrix.
    gpu_info_example = {
        "A10G": {
            "cost": 1.01,
            "tputs": [[5, 1], [10, 5]],
        },
        "A100": {
            "cost": 3.67,
            "tputs": [[20, 2], [50, 20]],
        },
    }

    # A 2D matrix obtained by analyzing the request distributions of the dataset.
    workload_distribution = [[0.25, 0.5], [0.25, 0.25]]

    # parameters
    # Overall rate represents the total number of requests per second of the workload.
    overall_rate = 16
    # Slice factor represents how many slices each bucket is split into.
    # A slice factor of 16 means that each bucket is split into 16 slices.
    slice_factor = 1

    #### Run the solver ####
    mix_result = HeteroAcceleratorSolver(
        workload_distribution=workload_distribution,
        overall_rate=overall_rate,
        slice_factor=slice_factor,
        gpu_info=gpu_info_example,
    ).run()
    print(mix_result)


if __name__ == "__main__":
    solver_run_example()
