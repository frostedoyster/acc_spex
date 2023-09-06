import torch
from acc_spex.lib import ref_spex, acc_spex
import time


def benchmark(dtype, device):

    n_species = 3
    n_nodes = 100
    n_edges = 6000
    n_max = [7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1]
    l_max = len(n_max) - 1

    print(f"Benchmarking dtype {dtype} and device {device}")
    centers = torch.randint(n_nodes, (n_edges,), device=device)
    centers = torch.sort(centers)[0]
    neighbors = torch.randint(n_nodes, (n_edges,), device=device)
    radial_features = [torch.rand(n_edges, n_max[l], dtype=dtype, device=device) for l in range(l_max+1)]
    angular_features = [torch.rand(n_edges, 2*l+1, dtype=dtype, device=device) for l in range(l_max+1)]
    node_species = torch.randint(n_species, (n_nodes,), device=device)

    for _ in range(10):
        ref_spex(centers, neighbors, radial_features, angular_features, node_species, n_species)

    start = time.time()
    for _ in range(1000):
        ref_spex(centers, neighbors, radial_features, angular_features, node_species, n_species)
    if device == "cuda": torch.cuda.synchronize()
    finish = time.time()
    print(f"The pure torch implementation fwd took {finish-start} seconds")

    '''
    start = time.time()
    for _ in range(1000):
        loss = torch.sum(ref_spex(a, b, indices, nnodes))
        loss.backward()
    torch.cuda.synchronize()
    finish = time.time()
    print(
        f"The pure torch implementation fwd + bwd took {finish-start} seconds")
    '''

    for _ in range(10):
        acc_spex(centers, neighbors, radial_features, angular_features, node_species, n_species)

    start = time.time()
    for _ in range(1000):
        acc_spex(centers, neighbors, radial_features, angular_features, node_species, n_species)
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The accelerated implementation fwd took {finish-start} seconds")

    '''
    start = time.time()
    for _ in range(1000):
        loss = torch.sum(acc_spex(a, b, indices, nnodes))
        loss.backward()
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The accelerated implementation fwd + bwd took {finish-start} seconds")
    '''


if __name__ == "__main__":
    benchmark(torch.float32, "cpu")
    benchmark(torch.float64, "cpu")
    """
    if torch.cuda.is_available():
        benchmark(torch.float32, "cuda")
        benchmark(torch.float64, "cuda")
     """
