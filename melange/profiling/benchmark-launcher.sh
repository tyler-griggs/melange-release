#!/bin/bash

# Result files will be added to 'PATH_PREFIX' directory.
PATH_PREFIX='results'

# TODO: Set your preferred request sizes and rates here.
for input_len in 25 100 250 500 1000 2000; do
  for output_len in 25 100 250 500 1000 2000; do
    for req_rate in 1 2 4 8 16 32; do
      OUTPUT_FILE="${PATH_PREFIX}/${input_len}-${output_len}-${req_rate}-${TOTAL}.txt"
      python gpu-benchmark.py --backend=vllm --tokenizer=hf-internal-testing/llama-tokenizer --dataset=junk --request-rate=$req_rate --num-prompts=$TOTAL --input_len $input_len --output_len $output_len > ${OUTPUT_FILE}
    done
  done
done
