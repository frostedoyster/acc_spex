from acc_spex.lib import ref_spex, acc_spex
import torch
torch.manual_seed(0)


def test(dtype, device):
    print(f"Testing dtype {dtype} and device {device}")

    n_species = 3
    n_nodes = 30
    n_edges = 1000
    n_max = [7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1]
    l_max = len(n_max) - 1

    print(f"Benchmarking dtype {dtype} and device {device}")
    centers = torch.randint(n_nodes, (n_edges,), device=device)
    centers = torch.sort(centers)[0]

    # create a missing center:
    where_20 = torch.where(centers == 20)[0]
    centers[where_20] = 19

    neighbors = torch.randint(n_nodes, (n_edges,), device=device)
    radial_features_ref = [torch.rand(n_edges, n_max[l], dtype=dtype, device=device, requires_grad=True) for l in range(l_max+1)]
    angular_features_ref = [torch.rand(n_edges, 2*l+1, dtype=dtype, device=device, requires_grad=True) for l in range(l_max+1)]
    radial_features_acc = [radial_features_ref[l].clone().detach() for l in range(l_max+1)]
    angular_features_acc = [angular_features_ref[l].clone().detach() for l in range(l_max+1)]
    for l in range(l_max+1):
        radial_features_acc[l].requires_grad_(True)
        angular_features_acc[l].requires_grad_(True)
    node_species = torch.randint(n_species, (n_nodes,), device=device)

    # create a missing species:
    where_1 = torch.where(node_species == 1)[0]
    node_species[where_1] = 2

    ref_spex_result = ref_spex(centers, neighbors, radial_features_ref, angular_features_ref, node_species, n_species)
    acc_spex_result = acc_spex(centers, neighbors, radial_features_acc, angular_features_acc, node_species, n_species)

    for l in range(l_max+1):
        assert torch.allclose(ref_spex_result[l], acc_spex_result[l])
 
    loss_ref = torch.sum(torch.stack([torch.sum(ref_spex_result[l]) for l in range(l_max+1)]))
    loss_ref.backward()
    loss_acc = torch.sum(torch.stack([torch.sum(acc_spex_result[l]) for l in range(l_max+1)]))
    loss_acc.backward()

    for l in range(l_max+1):
        assert torch.allclose(radial_features_ref[l].grad, radial_features_acc[l].grad)
        assert torch.allclose(angular_features_ref[l].grad, angular_features_acc[l].grad)    

    print("Assertions passed successfully!")


if __name__ == "__main__":
    test(torch.float32, "cpu")
    test(torch.float64, "cpu")
    """
    if torch.cuda.is_available():
        test(torch.float32, "cuda")
        test(torch.float64, "cuda")
    """
