# 在rdma0上创建2个VF
echo 0 | sudo tee /sys/class/net/rdma0/device/sriov_numvfs
echo 2 | sudo tee /sys/class/net/rdma0/device/sriov_numvfs
VF_A=$(basename /sys/class/net/rdma0/device/virtfn0/net/enp26s0f0v0)
VF_B=$(basename /sys/class/net/rdma0/device/virtfn1/net/enp26s0f0v1)
echo "VF_A=$VF_A  VF_B=$VF_B"

DEV_ARGS=()
for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm /dev/infiniband/umad*; do
  [ -e "$d" ] && DEV_ARGS+=( --device="$d" )
done

docker run -d --name node0 --network=none --ipc=host \
  --gpus all \
  --entrypoint sleep \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  "${DEV_ARGS[@]}" \
  -v ~/moe_test/vllm/profile_json:/profile_json \
  -v ~/moe_test/vllm/moe_test:/moe_test \
  soryxie/moe-profile:v0.10.0-rdma infinity

docker run -d --name node1 --network=none --ipc=host \
  --gpus all \
  --entrypoint sleep \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  "${DEV_ARGS[@]}" \
  -v ~/moe_test/vllm/profile_json:/profile_json \
  -v ~/moe_test/vllm/moe_test:/moe_test \
  soryxie/moe-profile:v0.10.0-rdma infinity


# Optional: PF 层面对两个 VF 开 trust、关 spoofchk，后续容器内改 MAC/VLAN 会更自由
sudo ip link set dev rdma0 vf 0 trust on
sudo ip link set dev rdma0 vf 1 trust on
sudo ip link set dev rdma0 vf 0 spoofchk off
sudo ip link set dev rdma0 vf 1 spoofchk off

PID0=$(docker inspect -f '{{.State.Pid}}' node0)
PID1=$(docker inspect -f '{{.State.Pid}}' node1)
echo "PID0=$PID0 PID1=$PID1"

# 把两个 VF 迁到各自容器 netns
sudo ip link set "$VF_A" netns "$PID0"
sudo ip link set "$VF_B" netns "$PID1"

# 在容器 netns 内改名/拉起/配 IP（RoCE/以太网示例网段 192.168.100.0/24）
sudo nsenter -t "$PID0" -n -- bash -lc "
ip link set dev $VF_A name ethvf0
ip link set ethvf0 up
ip addr add 192.168.100.2/24 dev ethvf0
"
sudo nsenter -t "$PID1" -n -- bash -lc "
ip link set dev $VF_B name ethvf1
ip link set ethvf1 up
ip addr add 192.168.100.3/24 dev ethvf1
"

# 连通性测试
sudo nsenter -t "$PID0" -n -- ping -c2 192.168.100.3

# 流量侦测结果：
# after : tx=492386 rx=486351
# delta : tx=491414 rx=486035

# node0
docker exec -it node0 bash -lc '
set -e
HCA=$(basename /sys/class/net/ethvf0/device/infiniband/*)
echo "node0: ethvf0 -> $HCA"
export NCCL_DEBUG=INFO
export NCCL_IB_HCA="$HCA"
export NCCL_IB_GID_INDEX=1          # cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/1 "RoCE v2"
export NCCL_SOCKET_IFNAME=ethvf0
export MASTER_ADDR=192.168.100.2    # rank0 的 IP（ethvf0）
export MASTER_PORT=12345


IF=ethvf0
TX_BEFORE=$(cat /sys/class/net/$IF/statistics/tx_bytes)
RX_BEFORE=$(cat /sys/class/net/$IF/statistics/rx_bytes)
echo "before: tx=$TX_BEFORE rx=$RX_BEFORE"


torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /moe_test/moe_8_card_profile.py

TX_AFTER=$(cat /sys/class/net/$IF/statistics/tx_bytes)
RX_AFTER=$(cat /sys/class/net/$IF/statistics/rx_bytes)
echo "after : tx=$TX_AFTER rx=$RX_AFTER"
echo "delta : tx=$((TX_AFTER-TX_BEFORE)) rx=$((RX_AFTER-RX_BEFORE))"
'

docker exec -it node1 bash -lc '
set -e
HCA=$(basename /sys/class/net/ethvf1/device/infiniband/*)
echo "node1 HCA=$HCA on ethvf1"
export NCCL_DEBUG=INFO
export NCCL_IB_HCA="$HCA"
export NCCL_IB_GID_INDEX=1
export NCCL_SOCKET_IFNAME=ethvf1
export MASTER_ADDR=192.168.100.2
export MASTER_PORT=12345

torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
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
