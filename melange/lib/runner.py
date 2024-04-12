from dataclasses import dataclass, field
import json
from typing import Dict
from ruamel.yaml import YAML
from pathlib import Path

from melange.lib.util import CloudInstanceInfo, get_instance_info
from melange.solver import MelangeSolver, Solver

PROJECT_DIR = Path(__file__).parent.parent.parent
OUTPUT_TEMPLATE_DIR = Path(__file__).parent.parent / "output"

class SolverRunner:
    @dataclass
    class Config:
        gpu_info: dict = field(default_factory=dict)
        workload_distribution: list = field(default_factory=list)
        total_request_rate: float = 0 # units: requests per second
        slice_factor: int = 1

    def __init__(self, config_path: str):
        self.config: SolverRunner.Config = SolverRunner.Config(**json.load(open(config_path)))
        self.solver: Solver = MelangeSolver(
            workload_distribution=self.config.workload_distribution,
            total_request_rate=self.config.total_request_rate,
            gpu_info=self.config.gpu_info,
            slice_factor=self.config.slice_factor
        )
        self.execution_result: Dict[str, CloudInstanceInfo] = {}

    def run(self):
        execution_result = self.solver.run()
        print(f"[Melange] Recommendation: {execution_result}")

        # cache the execution result
        for key, value in execution_result.items():
            if key == "cost" or value <= 0:
                continue
            self.execution_result[key] = get_instance_info(gpu_type=key, num_of_instance=value)

    def export(self, output_format: str):
        if output_format == "default":
            self._export_default()
        elif output_format == "skypilot":
            self._export_skypilot()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _export_default(self):
        output_path = PROJECT_DIR / "melange_result.json"
        execution_result = []
        for gpu_type, instance_info in self.execution_result.items():
            execution_result.append(instance_info.__dict__)
        with open(output_path, "w") as f:
            json.dump(execution_result, f, indent=4)

        print(f"[Melange] Exported results to: {output_path}")

    def _export_skypilot(self):
        template_path = OUTPUT_TEMPLATE_DIR / "skypilot_template.yaml"
        with open(template_path, "r") as f:
            yaml = YAML() # default uses round-trip loader/dumper to preserve formats
            template = yaml.load(f)

        for gpu_type, instance_info in self.execution_result.items():
            for instance_count in range(instance_info.instance_num):
                template["resources"]["cloud"] = instance_info.cloud
                template["resources"]["region"] = instance_info.region
                template["resources"]["instance_type"] = instance_info.instance_type

                output_path = PROJECT_DIR / f"skypilot_{gpu_type}_{instance_count}.yaml"
                with open(output_path, "w") as f:
                    yaml.dump(template, f)

            print(f"[Melange] Exported result for {gpu_type} to: {output_path}")
