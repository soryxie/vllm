
# https://github.com/NVIDIA/nvbandwidth
apt install libboost-program-options-dev

git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth
export PATH=$PATH:/usr/local/cuda/bin
cmake .
make

./nvbandwidth -t device_to_device_memcpy_read_ce


