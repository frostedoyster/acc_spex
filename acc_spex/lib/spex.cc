#include <iostream>
#include <torch/extension.h>
#include <omp.h>


torch::Tensor find_first_occurrences(torch::Tensor scatter_indices, long out_dim) {
    // Finds the positions of the first occurrences within scatter_indices
    
    long scatter_size = scatter_indices.size(0);
    long* scatter_indices_ptr = scatter_indices.data_ptr<long>();
    torch::Tensor first_occurrences = torch::empty({out_dim}, torch::dtype(torch::kLong));
    first_occurrences.fill_(-1);
    long* first_occurrences_ptr = first_occurrences.data_ptr<long>();
    first_occurrences_ptr[scatter_indices_ptr[0]] = 0;

    #pragma omp parallel for
    for (long i = 0; i < scatter_size-1; i++) {
        if (scatter_indices_ptr[i] < scatter_indices_ptr[i+1]) first_occurrences_ptr[scatter_indices_ptr[i+1]] = i+1;
    }
    if (first_occurrences_ptr[out_dim-1] == -1) first_occurrences_ptr[out_dim-1] = scatter_size;
    for (long i = out_dim - 2; i > -1; i--) {
        if (first_occurrences_ptr[i] == -1) first_occurrences_ptr[i] = first_occurrences_ptr[i+1];
    }

    return first_occurrences;
}


template<typename scalar_t>
torch::Tensor forward_t(
    torch::Tensor first_occurrences,
    torch::Tensor radial_features,
    torch::Tensor angular_features,
    torch::Tensor neighbor_species,
    long n_nodes,
    long n_species
) {

    long n_max_l = radial_features.size(1);
    long m_size = angular_features.size(1);
    long n_edges = neighbor_species.size(0);
    torch::Tensor result = torch::zeros(
        {n_nodes, m_size, n_species*n_max_l},
        torch::TensorOptions().device(radial_features.device()).dtype(radial_features.dtype())
    );

    scalar_t* result_ptr = result.data_ptr<scalar_t>();
    scalar_t* radial_features_ptr = radial_features.data_ptr<scalar_t>();
    scalar_t* angular_features_ptr = angular_features.data_ptr<scalar_t>();
    long* first_occurrences_ptr = first_occurrences.data_ptr<long>();
    long* neighbor_species_ptr = neighbor_species.data_ptr<long>();

    #pragma omp parallel for schedule(dynamic)
    for (long i_node = 0; i_node < n_nodes; i_node++) {
        long first_occurrence = first_occurrences_ptr[i_node];
        long last_occurrence;
        if (i_node == n_nodes - 1) {
            last_occurrence = n_edges;
        } else {
            last_occurrence = first_occurrences_ptr[i_node+1];
        }
        for (long i_edge = first_occurrence; i_edge < last_occurrence; i_edge++) {
            for (long m = 0; m < m_size; m++) {
                for (long n = 0; n < n_max_l; n++) {
                    result_ptr[i_node*m_size*n_species*n_max_l + m*n_species*n_max_l + neighbor_species_ptr[i_edge]*n_max_l + n]
                    += radial_features_ptr[i_edge*n_max_l + n] * angular_features_ptr[i_edge*m_size + m];
                }
            }
        }
    }

    return result;
}


class SpexL : public torch::autograd::Function<SpexL> {

public:

    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor first_occurrences,
        torch::Tensor radial_features,
        torch::Tensor angular_features,
        torch::Tensor neighbor_species,
        long n_nodes,
        long n_species
    ) {
        ctx->save_for_backward({first_occurrences, radial_features, angular_features, neighbor_species});
        ctx->saved_data["n_nodes"] = n_nodes;
        ctx->saved_data["n_species"] = n_species;

        // Dispatch type by hand
        if (radial_features.dtype() == c10::kDouble) {
            return forward_t<double>(first_occurrences, radial_features, angular_features, neighbor_species, n_nodes, n_species);
        } else if (radial_features.dtype() == c10::kFloat) {
            return forward_t<float>(first_occurrences, radial_features, angular_features, neighbor_species, n_nodes, n_species);
        } else {
            throw std::runtime_error("Unsupported dtype");
        }
    }

    static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outputs) {
        throw std::runtime_error("Not implemented");
    }
};


std::vector<torch::Tensor> spex(
    torch::Tensor centers,
    torch::Tensor neighbors,
    std::vector<torch::Tensor> radial_features,
    std::vector<torch::Tensor> angular_features,
    torch::Tensor node_species,
    long n_species
) {
    long n_nodes = node_species.size(0);
    torch::Tensor first_occurrences = find_first_occurrences(centers, n_nodes);
    torch::Tensor neighbor_species = node_species.index_select(0, neighbors);

    long l_max = radial_features.size() - 1;
    std::vector<torch::Tensor> spherical_expansion;
    spherical_expansion.reserve(l_max+1);
    for (long l = 0; l < l_max+1; l++) {
        spherical_expansion.push_back(
            SpexL::apply(first_occurrences, radial_features[l], angular_features[l], neighbor_species, n_nodes, n_species)
        );
    }
    return spherical_expansion;
}


TORCH_LIBRARY(spex_cc, m) {
    m.def("spex", &spex);
}

