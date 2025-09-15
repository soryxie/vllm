import os
import re
import json
import glob
import argparse
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt

def read_files(k=2, reuse=60, only_all2all=True):
    layer_times = []
    with open("./profile_json/compute.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            layer_times.append(data)

    all2all_times = []
    with open(f"./profile_json/K_{k}_REUSE_{reuse}_{'ALL2ALL' if only_all2all else 'ALLFLOW'}.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            all2all_times.append(data)

    return layer_times, all2all_times

def main():
    for k in [2, 4]:
        for reuse in [60, 80]:
            all2all_time_cache = None
            for only_all2all in [True, False]:
                layer_times, all2all_times = read_files(k=k, reuse=reuse, only_all2all=only_all2all)

                layer_time_0 = layer_times[0]
                all2all_times = all2all_times[1:]
                attn, mlp, native_comm = layer_time_0["attn"], layer_time_0["mlp"], layer_time_0["all2all"]
                all2all = []
                for all2all_time in all2all_times:
                    all2all.append(all2all_time["time"])
                all2all = sum(all2all) / len(all2all_times)

                colors = ['#8dd3c7', '#bebada', '#fb8072']
                plt.figure()
                plt.pie([attn, mlp, all2all] \
                    , labels=[f'Attention {attn:.4f}', f'MLP {mlp:.4f}', f'All2All {all2all:.4f}']
                    , autopct='%1.1f%%', startangle=90, colors=colors)
                plt.title(f'Time breakdown K={k} Reuse={reuse} {"All2All" if only_all2all else "AllFlow"}')
                plt.axis('equal')
                plt.savefig(f"time_breakdown_K_{k}_REUSE_{reuse}_{'ALL2ALL' if only_all2all else 'ALLFLOW'}.png")

                if only_all2all:
                    all2all_time_cache = all2all
                else:
                    print(f"K={k} Reuse={reuse} AllFlow Enlarge All2All time from {all2all_time_cache:.4f} to {all2all:.4f}, ratio: {all2all/all2all_time_cache:.4f}")


if __name__ == "__main__":
    main()
