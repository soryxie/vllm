import os
import re
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_files():
    layer_times = []
    with open("./profile_json/compute.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            layer_times.append(data)

    all2all_times = []
    with open("./profile_json/moe_profile_8card.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            all2all_times.append(data)

    return layer_times, all2all_times

def main():
    layer_times, all2all_times = read_files()

    i = 0
    attn = []
    mlp = []
    all2all_comm = []
    native_comm = []
    for i, layer, all2all in zip(range(len(layer_times)), layer_times, all2all_times):
        attn.append(layer["attn"])
        mlp.append(layer["mlp"])
        native_comm.append(layer["all2all"])
        all2all_comm.append(all2all["time_ms"])

    attn_sum = sum(attn)
    mlp_sum = sum(mlp)
    native_comm_sum = sum(native_comm)
    all2all_comm_sum = sum(all2all_comm) / 2

    colors = ['#8dd3c7', '#bebada', '#fb8072']
    plt.pie([attn_sum, mlp_sum, all2all_comm_sum] \
        , labels=[f'Attention {attn_sum*1000:.2f}ms', f'MLP {mlp_sum*1000:.2f}ms', f'All2All {all2all_comm_sum*1000:.2f}ms']
        , autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Time breakdown')
    plt.axis('equal')
    plt.savefig('time_breakdown.png')
    print(f"all2all to native_comm ratio: {all2all_comm_sum / native_comm_sum:.2f}")

if __name__ == "__main__":
    main()
