export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3 
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 8848 \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --max_model_len 4096

# env:
# pip install --upgrade uv
# uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
# python use_existing_torch.py
# uv pip install -r requirements/build.txt
# uv pip install --no-build-isolation -e .