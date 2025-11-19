export NCCL_DEBUG=INFO
export HF_HOME=/mnt/nfs/songrun/huggingface
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3 
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 8848 \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --enforce-eager


python3 benchmarks/benchmark_serving.py \
    --base-url http://localhost:8848 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1 \
    --num-prompts 4 \
    --save-result \
    --result-dir ./profile_json/ \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507

# env:
# pip install --upgrade uv
# uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
# python use_existing_torch.py
# uv pip install -r requirements/build.txt
# uv pip install --no-build-isolation -e . 

docker run --name songrun_ep --gpus all -it --volume /home/songrun/:/usr/wkspace \
    --volume /mnt/nfs/songrun:/usr/data \
    --privileged --device="/dev/infiniband/*" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint /bin/bash \
    --network=host vllm/vllm-openai:v0.9.2

# gpu11
LMCACHE_CONFIG_FILE="mooncake-config.yaml" \
LMCACHE_USE_EXPERIMENTAL=True \
VLLM_TORCH_PROFILER_DIR=/usr/wkspace/vllm_songrun \
HF_HOME=/mnt/nfs/songrun/.cache/huggingface/ \
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
            --port 8200 \
            --disable-log-requests \
            --max-num-seqs 4 \
            --enforce-eager \
            --no-enable-prefix-caching \
            --tensor-parallel-size 1 \
            --data-parallel-size 4 \
            --enable-expert-parallel \
            --max_model_len 4096 \
            --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'

# gpu12
LMCACHE_CONFIG_FILE="mooncake-config.yaml" \
LMCACHE_USE_EXPERIMENTAL=True \
VLLM_TORCH_PROFILER_DIR=/usr/wkspace/vllm_songrun \
HF_HOME=/usr/wkspace/.cache/huggingface/ \
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
            --port 8200 \
            --disable-log-requests \
            --max-num-seqs 4 \
            --enforce-eager \
            --no-enable-prefix-caching \
            --tensor-parallel-size 1 \
            --data-parallel-size 4 \
            --enable-expert-parallel \
            --max_model_len 4096 \
            --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'

# --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}'
#

VLLM_TORCH_PROFILER_DIR=/usr/wkspace/vllm_songrun \
HF_HOME=/usr/wkspace/.cache/huggingface/ \
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
            --port 8200 \
            --disable-log-requests \
            --max-num-seqs 4 \
            --enforce-eager \
            --no-enable-prefix-caching \
            --tensor-parallel-size 1 \
            --data-parallel-size 4 \
            --enable-expert-parallel \
            --max_model_len 4096
