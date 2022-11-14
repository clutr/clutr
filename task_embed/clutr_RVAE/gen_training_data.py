
import random
from random import choices
def generate_minigrid_data(grid_size=169, max_seq_len=52, num_samples=10000, filename='minigrid_train', chunk_size=32):
    seq_set = set()
    base = list(range(1, grid_size + 1))
    seqs = []

    while len(seqs) < num_samples:
        for ln in range(2, max_seq_len + 1):
            for _ in range(chunk_size):
                sub = random.sample(base, ln)
                walls = sub[0:-2]
                walls.sort()
                sub = walls + sub[-2:]
                if not tuple(sub) in seq_set:
                    seq_set.add(tuple(sub))
                    seqs.append(sub)

    #seqs = [list(s) for s in seqs]

    flat_str = lambda lst: " ".join([str(l) for l in lst])
    s = ""
    for seq in seqs:
        s += (flat_str(seq) + "\n")

    with open(f"{filename}_{num_samples}_{chunk_size}.txt", "w+") as f:
        f.write(s)



if __name__ == "__main__":
    
    print("Generating Data")
    generate_minigrid_data(num_samples=10000000)
