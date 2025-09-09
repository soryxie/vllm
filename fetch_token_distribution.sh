sudo docker run --rm -it --gpus all --network=host \
    --entrypoint /bin/bash \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/moe_test/vllm/profile_json:/profile_json \
    -p 8000:8000 \
    --ipc=host \
    soryxie/moe-profile:v0.10.0

# for tokens in each node = 1024/2048, change: max_num_batched_tokens, dataset length
# vim vllm/model_executor/layers/fused_moe/layer.py
# vim benchmarks/benchmark_dataset.py

export CUDA_VISIBLE_DEVICES=0,1,2,3; python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 8001 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 1025 \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 1024 \
    --data-parallel-size 4 \
    --max_num_seqs=1 \
    --enforce-eager > /profile_json/log.txt 2>&1 &

# benchmark
pip install datasets pandas
python3 benchmarks/benchmark_serving.py \
    --base-url http://localhost:8001 \
    --dataset-name sharegpt \
    --dataset-path /profile_json/ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-output-len 1 \
    --max_concurrency 1 \
    --num-prompts 8 \
    --save-result \
    --save-detailed \
    --result-dir /profile_json/ \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --request-rate inf
