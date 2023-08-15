import torch


class LEIterator(torch.nn.Module):

    def __init__(self, combine_indices, multiplicities):
        super().__init__()

        self.combine_indices = combine_indices
        self.multiplicities = multiplicities
        self.nu_max = len(combine_indices) - 1

    def forward(self, LE_1):
        # LE_1 is in format (l) -> [b, m, i]

        LE_features = [0, LE_1]
        for nu in range(2, self.nu_max+1):
            LE_features.append({})
            LE_features_nu_minus_one = LE_features[nu-1]
            for l_tuple_nu, combine_indices_l_tuple_nu in self.combine_indices[nu].items():
                l_tuple_nu_minus_one = l_tuple_nu[:-1]
                l_nu = l_tuple_nu[-1]
                LE_features_nu_minus_one_l = LE_features_nu_minus_one[l_tuple_nu_minus_one]
                LE_features_1_l = LE_1[(l_nu,)]
                indices_nu_minus_one_l = combine_indices_l_tuple_nu[0]
                indices_1_l = combine_indices_l_tuple_nu[1]
                LE_features[nu][l_tuple_nu] = LE_features_nu_minus_one_l.index_select(0, indices_nu_minus_one_l).unsqueeze(2)*LE_features_1_l.index_select(0, indices_1_l).unsqueeze(1)
                LE_features[nu][l_tuple_nu] = LE_features[nu][l_tuple_nu].reshape(LE_features[nu][l_tuple_nu].shape[0], -1, LE_features[nu][l_tuple_nu].shape[3])
        
        # Apply the multiplicities:
        for nu in range(2, self.nu_max+1):
            for l_tuple_nu, features in LE_features[nu].items():
                features *= self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2)

        return LE_features


    def compute_with_gradients(self, LE_1_values, LE_1_gradients, gradient_metadata):
        # LE_1_values is in format (l) -> [b, m, i]
        # LE_1_gradients is in format (l) -> [b, m, ij, alpha]

        LE_values = [0, LE_1_values]
        LE_gradients = [0, LE_1_gradients]
        for nu in range(2, self.nu_max+1):
            LE_values.append({})
            LE_gradients.append({})
            LE_values_nu_minus_one = LE_values[nu-1]
            LE_gradients_nu_minus_one = LE_gradients[nu-1]
            for l_tuple_nu, combine_indices_l_tuple_nu in self.combine_indices[nu].items():
                l_tuple_nu_minus_one = l_tuple_nu[:-1]
                l_nu = l_tuple_nu[-1]
                LE_values_nu_minus_one_l = LE_values_nu_minus_one[l_tuple_nu_minus_one]
                LE_values_1_l = LE_1_values[(l_nu,)]
                LE_gradients_nu_minus_one_l = LE_gradients_nu_minus_one[l_tuple_nu_minus_one]
                LE_gradients_1_l = LE_1_gradients[(l_nu,)]
                indices_nu_minus_one_l = combine_indices_l_tuple_nu[0]
                indices_1_l = combine_indices_l_tuple_nu[1]
                LE_values[nu][l_tuple_nu] = LE_values_nu_minus_one_l.index_select(0, indices_nu_minus_one_l).unsqueeze(2)*LE_values_1_l.index_select(0, indices_1_l).unsqueeze(1)
                LE_gradients[nu][l_tuple_nu] = (
                    LE_values_nu_minus_one_l.index_select(0, indices_nu_minus_one_l).index_select(2, gradient_metadata).unsqueeze(2).unsqueeze(4)*LE_gradients_1_l.index_select(0, indices_1_l).unsqueeze(1)
                    +
                    LE_gradients_nu_minus_one_l.index_select(0, indices_nu_minus_one_l).unsqueeze(2)*LE_values_1_l.index_select(0, indices_1_l).index_select(2, gradient_metadata).unsqueeze(1).unsqueeze(4)
                )
                LE_values[nu][l_tuple_nu] = LE_values[nu][l_tuple_nu].reshape(LE_values[nu][l_tuple_nu].shape[0], -1, LE_values[nu][l_tuple_nu].shape[3])
                LE_gradients[nu][l_tuple_nu] = LE_gradients[nu][l_tuple_nu].reshape(LE_gradients[nu][l_tuple_nu].shape[0], -1, LE_gradients[nu][l_tuple_nu].shape[3], 3)
        
        # Apply the multiplicities:
        for nu in range(2, self.nu_max+1):
            for l_tuple_nu, values in LE_values[nu].items():
                values *= self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2)
            for l_tuple_nu, gradients in LE_gradients[nu].items():
                gradients *= self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        return LE_values, LE_gradients

            


