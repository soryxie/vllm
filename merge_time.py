import json

layer_time = []
with open("profile_json/layer_time.jsonl", "r") as f:
    for line in f:
        layer_time.append(json.loads(line))

all2all_time = []
with open("profile_json/all2all_time.jsonl", "r") as f:
    for line in f:
        all2all_time.append(json.loads(line))

new_layer_time = []

for i, item in enumerate(layer_time):
    t0 = item["st"]
    t1 = all2all_time[i*2]["st"]
    t2 = all2all_time[i*2]["et"]
    t3 = all2all_time[i*2+1]["st"]
    t4 = all2all_time[i*2+1]["et"]
    t5 = item["et"]
    assert t0 <= t1 <= t2 <= t3 <= t4 <= t5

    all2all = (t2 - t1) + (t4 - t3)
    mlp = t3 - t2
    attn = (t5 - t0) - all2all - mlp

    if item["batch_size"] == 1024:
        new_layer_time.append({
            "attn": attn,
            "mlp": mlp,
            "all2all": all2all,
        })

print(len(new_layer_time))

final_layer_time = []
for i in range(48):
    t = [
        new_layer_time[i + j*48] for j in range(8)
    ]
    t = t[1:]  
    final_layer_time.append({
        "layer": i,
        "attn": sum(x["attn"] for x in t) / 7,
        "mlp": sum(x["mlp"] for x in t) / 7,
        "all2all": sum(x["all2all"] for x in t) / 7,
    })

with open("profile_json/merge_time.jsonl", "w") as f:
    for item in final_layer_time:
        f.write(json.dumps(item) + "\n")