sudo docker run --rm -it --gpus all --network=host \
    --entrypoint /bin/bash \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /home/songrun/vllm/profile_json:/profile_json \
    -p 8001:8000 \
    --ipc=host \
    soryxie/moe-profile:v0.10.0

# vim vllm/distributed/device_communicators/all2all.py
# vim vllm/model_executor/layers/fused_moe/layer.py
# vim vllm/model_executor/models/qwen3_moe.py

export CUDA_VISIBLE_DEVICES=0,1,2,3; python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 1025 \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 1024 \
    --data-parallel-size 4 \
    --max_num_seqs=1 \
    --port 8001 \
    --enforce-eager > /profile_json/log.txt 2>&1 &

pip install datasets pandas
python3 benchmarks/benchmark_serving.py \
    --base-url http://localhost:8001 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1 \
    --num-prompts 1 \
    --save-result \
    --save-detailed \
    --result-dir /profile_json/ \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507

# remove the first 96 lines of the jsonl files (warmup)
tail -n +97 profile_json/all2all_time.jsonl > profile_json/temp.jsonl && mv profile_json/temp.jsonl profile_json/all2all_time.jsonl
tail -n +97 profile_json/layer_time.jsonl > profile_json/temp.jsonl && mv profile_json/temp.jsonl profile_json/layer_time.jsonl
