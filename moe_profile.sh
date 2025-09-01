sudo docker run --rm -it --gpus all \
    --entrypoint /bin/bash \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/moe_test/vllm/profile_json:/profile_json \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8001:8000 \
    --ipc=host \
    soryxie/moe-profile:v0.10.0

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --enable-expert-parallel \
    --data-parallel-size 4 \
    --enforce-eager > /profile_json/log.txt 2>&1 &

curl -X POST -s http://localhost:8001/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}'

# vim vllm/distributed/device_communicators/all2all.py
# vim vllm/model_executor/layers/fused_moe/layer.py

# vim vllm/model_executor/models/qwen3_moe.py
# vim vllm/distributed/device_communicators/all2all.py

pip install datasets pandas
python3 benchmarks/benchmark_serving.py \
    --base-url http://localhost:8000 \
    --dataset-name sharegpt \
    --dataset-path /profile_json/ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-output-len 1 \
    --num-prompts 10 \
    --result-dir /profile_json/ \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --request-rate inf