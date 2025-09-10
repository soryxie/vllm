DEV_ARGS=()
for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm /dev/infiniband/umad*; do
  [ -e "$d" ] && DEV_ARGS+=( --device="$d" )
done

docker run -d --name node0 --network=host --ipc=host \
  --gpus all \
  --entrypoint sleep \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  "${DEV_ARGS[@]}" \
  -v ~/moe_test/vllm/profile_json:/profile_json \
  -v ~/moe_test/vllm/moe_test:/moe_test \
  soryxie/moe-profile:v0.10.0-rdma infinity

docker run -d --name node1 --network=host --ipc=host \
  --gpus all \
  --entrypoint sleep \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  "${DEV_ARGS[@]}" \
  -v ~/moe_test/vllm/profile_json:/profile_json \
  -v ~/moe_test/vllm/moe_test:/moe_test \
  soryxie/moe-profile:v0.10.0-rdma infinity

# 流量侦测结果：
# after : tx=492386 rx=486351
# delta : tx=491414 rx=486035

# node0
docker exec -it node0 bash -lc '
set -e
echo "node0 -> mlx5_0"
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3          # cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/1 "RoCE v2"
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=0,2

torchrun --nnodes=4 --nproc_per_node=2 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /moe_test/moe_8_card_profile.py
'

# node 1
docker exec -it node1 bash -lc '
set -e
echo "node1 -> mlx5_2"
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_2
export NCCL_IB_GID_INDEX=3
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=1,3

torchrun --nnodes=4 --nproc_per_node=2 --node_rank=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /moe_test/moe_8_card_profile.py
'

# node 2
docker exec -it node0 bash -lc '
set -e
echo "node0 -> mlx5_0"
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3 
export MASTER_ADDR=192.168.1.48
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=0,2

torchrun --nnodes=4 --nproc_per_node=2 --node_rank=2 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /moe_test/moe_8_card_profile.py
'

# node 3
docker exec -it node1 bash -lc '
set -e
echo "node1 -> mlx5_2"
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_2
export NCCL_IB_GID_INDEX=3
export MASTER_ADDR=192.168.1.48
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=1,3

torchrun --nnodes=4 --nproc_per_node=2 --node_rank=3 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /moe_test/moe_8_card_profile.py
'


# Appendix: 观察ip
PID0=$(docker inspect -f '{{.State.Pid}}' node0)
sudo nsenter -t "$PID0" -n ip -br addr show ethvf0

PID1=$(docker inspect -f '{{.State.Pid}}' node1)
sudo nsenter -t "$PID1" -n ip -br addr show ethvf1

# Cleanup
sudo docker stop node0 node1
sudo docker rm node0 node1

docker exec -it node0 bash -lc '
ib_write_bw -R -d mlx5_0 -F -p 19999 --report_gbits
' 

docker exec -it node1 bash -lc '
ib_write_bw -R -d mlx5_2 -F -p 19999 --report_gbits 10.0.12.1
'

sudo ip route add 10.2.11.0/24 via 10.0.12.254 dev rdma0
sudo ip route add 10.0.11.0/24 via 10.2.12.254 dev rdma2

docker exec -it node1 bash -lc '
ip addr show mlx5_0
'
