import json

from matplotlib.pylab import f


data = json.load(open("profile_json/logs/rank0.1758175173925841385.pt.trace.json"))
# data = json.load(open("profile_json/logs/rank0.1758166402953461720.pt.trace.json"))

Kernel_data = {}
Anon_cat = [
    'nccl:all_to_all',
    'p2d_batch_launch',
    'reuse_batch_launch',
    'other_0_batch_launch', 
    'other_1_batch_launch', 
    'other_2_batch_launch', 
    'other_3_batch_launch',
]
Anon_time_range = {cat: [] for cat in Anon_cat}
correlations = {}

for line in data["traceEvents"]:
    if "name" in line and "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)" in line["name"]:
        stream = line['args']['stream']
        if stream in Kernel_data:
            Kernel_data[stream].append(line)
        else:
            Kernel_data[stream] = [line]
        correlations[line['args']['correlation']] = {"stream": stream, "idx": len(Kernel_data[stream]) - 1}

    if "name" in line and "cuLaunchKernelEx" in line["name"]:
        if line['args']['correlation'] in correlations:
            cor = correlations[line['args']['correlation']]
            stream = cor["stream"]
            idx = cor["idx"]
            Kernel_data[stream][idx]["launch"] = line

    if "cat" in line and line["cat"] == "user_annotation":
        name = line["name"]
        if name in Anon_cat:
            start = line["ts"]
            end = start + line["dur"]
            Anon_time_range[name].append((start, end))

final_data = {}
for stream, kernels in Kernel_data.items():
    s_cat = None
    for kernel in kernels:
        if "launch" in kernel:
            launch = kernel["launch"]
            start = launch["ts"]
            end = start + launch["dur"]
            curr_cat = None
            for cat, ranges in Anon_time_range.items():
                for r in ranges:
                    if start >= r[0] and end <= r[1]:
                        found = True
                        curr_cat = cat
                        break
            assert curr_cat is not None, f"Error: cannot find cat for kernel {kernel}"
            assert s_cat is None or s_cat == curr_cat, f"Error: stream {stream} has mixed cat {s_cat} and {curr_cat}"
            s_cat = curr_cat
    # print(f"Stream {stream} has {len(kernels)} kernels, cat: {s_cat}")
    final_data[s_cat] = kernels.copy()

for cat, kernels in final_data.items():
    durations = [kernel["dur"] for kernel in kernels]
    # remove outliers, i.e., duration > mean + 2 * std
    mean = sum(durations) / len(durations)
    std = (sum((x - mean) ** 2 for x in durations) / len(durations)) ** 0.5
    durations = [x for x in durations if x <= mean + 2 * std and x >= mean - 2 * std]
    print(f"{cat.replace("_batch_launch", "")} \t len: {len(durations)} \t avg time: {sum(durations)/len(durations)/1000:.4f} ms")
