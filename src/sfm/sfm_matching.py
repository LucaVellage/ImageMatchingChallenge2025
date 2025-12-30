import path 
import numpy as np

def write_pairs_from_knn(
    pair_npz: Path,
    image_names: list[str],
    out_pairs_txt: Path,
    mutual: bool = True,
):
    """
    Convert pair_topk_K*.npz into COLMAP pairs.txt

    pair_npz:
        contains indices of top-K neighbors
    image_names:
        index -> filename mapping (order must match npz)
    """

    data = np.load(pair_npz)
    knn = data["pairs"] if "pairs" in data else data["indices"]

    edges = set()

    for i, nbrs in enumerate(knn):
        for j in nbrs:
            if i == j:
                continue
            if mutual:
                if i in knn[j]:
                    edges.add((i, j))
            else:
                edges.add((i, j))

    with open(out_pairs_txt, "w") as f:
        for i, j in sorted(edges):
            f.write(f"{image_names[i]} {image_names[j]}\n")
