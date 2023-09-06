import os
import sysconfig
import torch


import os
import sysconfig
import torch

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')

torch.ops.load_library(_HERE + '/spex_cc.so')
# if torch.cuda.is_available():
#    torch.ops.load_library(_HERE + '/spex_cu.so')


def ref_spex(
    centers,  # (edge,)  can be assumed to be ordered
    neighbors,  # (edge,)
    radial_features,  # l -> (edge, n) with n varying with l
    angular_features,  # l -> (edge, m)
    node_species,  # (node,)
    n_species  # int
):
    # Reference implementation

    n_nodes = node_species.shape[0]
    l_max = len(angular_features) - 1
    neighbor_species = node_species[neighbors]
    indices_for_index_add = centers*n_species+neighbor_species

    spherical_expansion = []
    for l in range(l_max+1):
        n_max_l = radial_features[l].shape[1]
        vector_expansion = angular_features[l].unsqueeze(2) * radial_features[l].unsqueeze(1)
        spherical_expansion_l = torch.zeros(n_nodes*n_species, 2*l+1, n_max_l, dtype=vector_expansion.dtype, device=vector_expansion.device)
        spherical_expansion_l.index_add_(0, indices_for_index_add, vector_expansion)
        spherical_expansion_l = spherical_expansion_l.reshape(n_nodes, n_species, 2*l+1, n_max_l).swapaxes(1, 2).reshape(n_nodes, 2*l+1, n_species*n_max_l).contiguous()
        spherical_expansion.append(spherical_expansion_l)

    return spherical_expansion


def acc_spex(
    centers,  # (edge,)  can be assumed to be ordered
    neighbors,  # (edge,)
    radial_features,  # l -> (edge, n) with n varying with l
    angular_features,  # l -> (edge, m)
    node_species,  # (node,)
    n_species  # int
):
    if centers.is_cuda:
        raise NotImplementedError()
    else:
        result = torch.ops.spex_cc.spex(
            centers.contiguous(),
            neighbors.contiguous(),
            [radial_features_l.contiguous() for radial_features_l in radial_features],
            [angular_features_l.contiguous() for angular_features_l in angular_features],
            node_species.contiguous(),
            n_species
        )

    return result
