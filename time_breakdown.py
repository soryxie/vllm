import os
import re
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_files():
    layer_times = []
    with open("./profile_json/layer_time.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            layer_times.append(data)
    all2all_times = []
    with open("./profile_json/all2all_time.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            all2all_times.append(data)
    
    with open("profile_json/moe_profile_8card_0.jsonl", "r") as f:
        lines = f.readlines()
        moe_profiles = []
        for line in lines:
            data = json.loads(line.strip())
            moe_profiles.append(data)

    return layer_times, all2all_times, moe_profiles

def main():
    layer_times, all2all_times, moe_profiles = read_files()

    native_comm = []
    all2all_comm = []
    attn = []
    mlp = []

    i = 0
    for layer, native, all2all in zip(layer_times, all2all_times, moe_profiles):
        attention_time = native["st"] - layer["st"]
        assert attention_time >= 0, f"layer {i}, native_st {native['st']}, layer_st {layer['st']}"

        mlp_time = layer["et"] - native["et"]
        assert mlp_time >= 0, f"layer {i}, native_et {native['et']}, layer_et {layer['et']}"

        native_comm.append(native["et"] - native["st"])
        all2all_comm.append(all2all["time_ms"])
        attn.append(attention_time)
        mlp.append(mlp_time)

        i += 1

    attn_avg = np.mean(attn)
    mlp_avg = np.mean(mlp)
    native_comm_avg = np.mean(native_comm)
    all2all_comm_avg = np.mean(all2all_comm)

    # mlp_avg -= all2all_comm_avg
    # all2all_comm_avg *= 2

    colors = ['#8dd3c7', '#bebada', '#fb8072']
    plt.pie([attn_avg, mlp_avg, all2all_comm_avg] \
        , labels=[f'Attention {attn_avg*1000:.2f}ms', f'MLP {mlp_avg*1000:.2f}ms', f'All2All {all2all_comm_avg*1000:.2f}ms']
        , autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Time breakdown')
    plt.axis('equal')
    plt.savefig('time_breakdown.png')
    print(f"all2all to native_comm ratio: {all2all_comm_avg / native_comm_avg:.2f}")

if __name__ == "__main__":
    main()
