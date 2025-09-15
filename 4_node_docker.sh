DEV_ARGS=()
for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm /dev/infiniband/umad*; do
  [ -e "$d" ] && DEV_ARGS+=( --device="$d" )
done

docker run -d --name node0 --network=host --ipc=host \
  --gpus all \
  --entrypoint sleep \
  --cap-add=IPC_LOCK --shm-size=48g --ulimit memlock=-1 \
  "${DEV_ARGS[@]}" \
  -v ~/moe_test/vllm/profile_json:/profile_json \
  -v ~/moe_test/vllm/moe_test:/moe_test \
  soryxie/moe-profile:v0.10.0-rdma infinity

docker run -d --name node1 --network=host --ipc=host \
  --gpus all \
  --entrypoint sleep \
  --cap-add=IPC_LOCK --shm-size=48g --ulimit memlock=-1 \
  "${DEV_ARGS[@]}" \
  -v ~/moe_test/vllm/profile_json:/profile_json \
  -v ~/moe_test/vllm/moe_test:/moe_test \
  soryxie/moe-profile:v0.10.0-rdma infinity

# node0
docker exec -it node0 bash -lc '
set -e
echo "node0 -> mlx5_0" 
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=4 --nproc_per_node=2 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /moe_test/moe_8_card_profile.py
'

# node 1
docker exec -it node0 bash -lc '
set -e
echo "node0 -> mlx5_0"
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3 
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=192.168.1.48
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=0,1

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
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=192.168.1.48
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=4 --nproc_per_node=2 --node_rank=2 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /moe_test/moe_8_card_profile.py
'

# node 3
docker exec -it node0 bash -lc '
set -e
echo "node0 -> mlx5_0"
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3 
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=192.168.1.48
export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=4 --nproc_per_node=2 --node_rank=3 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  /moe_test/moe_8_card_profile.py
'

# Cleanup
sudo docker stop node0 node1
sudo docker start node0 node1

# rdma profile
docker exec -it node0 bash -lc '
ib_write_bw -R -d mlx5_0 -F -p 19999 --report_gbits
' 

docker exec -it node1 bash -lc '
ib_write_bw -R -d mlx5_2 -F -p 19999 --report_gbits 10.0.12.1
'

# static route for 4 nodes
sudo ip route add 10.2.11.0/24 via 10.0.12.254 dev rdma0
sudo ip route add 10.0.11.0/24 via 10.2.12.254 dev rdma2
sudo ip route add 10.2.7.0/24 via 10.0.12.254 dev rdma0
sudo ip route add 10.0.7.0/24 via 10.2.12.254 dev rdma2
sudo ip route add 10.2.10.0/24 via 10.0.12.254 dev rdma0
sudo ip route add 10.0.10.0/24 via 10.2.12.254 dev rdma2

# cuda driver upgrade
# Removing nvidia-driver-550 (550.144.03-0ubuntu0.22.04.1) 
sudo apt-get purge -y nvidia-driver-550
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
sudo apt-get autoremove
sudo apt-get autoclean

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

sudo apt-get install -y cuda-drivers

# back to 550
sudo apt-get purge -y nvidia-driver-570
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
sudo apt-get autoremove
sudo apt-get autoclean

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get -y install cuda-toolkit-12-4

sudo apt-get install -y cuda-drivers-550
