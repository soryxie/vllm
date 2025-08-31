import os
import re
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, default="moe_profile_rank*.jsonl",
                    help="input file prefix (default: moe_profile_rank*.jsonl)")
    ap.add_argument("--out", type=str, default="tokens_send_recv.png",
                    help="output file name (default: tokens_send_recv.png)")
    return ap.parse_args()


def infer_rank_from_filename(path):
    m = re.search(r"rank(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    files = sorted(glob.glob(args.inputs))
    if not files:
        raise FileNotFoundError(f"File not found: {args.inputs}")

    total_recv = {}
    total_send = {}

    for fp in files:
        fn_rank = infer_rank_from_filename(fp)
        with open(fp, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rank = rec.get("rank", fn_rank)
                recv_counts = rec.get("recv_counts", [])
                send_counts = rec.get("send_counts", [])
                if rank is None:
                    continue
                if recv_counts:
                    total_recv[rank] = total_recv.get(rank, 0) + sum(int(x) for x in recv_counts)
                if send_counts:
                    total_send[rank] = total_send.get(rank, 0) + sum(int(x) for x in send_counts)

    ranks = sorted(set(total_recv.keys()) | set(total_send.keys()))
    recv_vals = [total_recv.get(r, 0) for r in ranks]
    send_vals = [total_send.get(r, 0) for r in ranks]

    x = np.arange(len(ranks))
    width = 0.35
    plt.figure(figsize=(max(6, len(ranks)*0.7), 5))
    plt.bar(x - width/2, recv_vals, width, label="Recv")
    plt.bar(x + width/2, send_vals, width, label="Send")
    plt.xticks(x, [f"GPU {r}" for r in ranks])
    plt.ylabel("Total tokens")
    plt.title("Total tokens received/sent per GPU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    plt.close()

    for r in ranks:
        print(f"GPU {r}: recv={total_recv.get(r, 0)}, send={total_send.get(r, 0)}")


if __name__ == "__main__":
    main()
