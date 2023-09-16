import torch
from .structures import transform_structures
from .LE_expansion import get_LE_expansion
from .polyeval import polyeval


class ACEEvaluator(torch.nn.Module):

    def __init__(
        self,
        all_species,
        composition_calculator,
        radial_spectrum_calculator,
        spherical_expansion_calculator,
        combine_indices,
        multipliers,
        l_max,
        E_nl,
        E_n0,
        E_max,
        device
    ):
        super().__init__()
        self.all_species = all_species
        self.composition_calculator = composition_calculator
        self.radial_spectrum_calculator = radial_spectrum_calculator
        self.spherical_expansion_calculator = spherical_expansion_calculator
        self.combine_indices = combine_indices
        self.multipliers = multipliers
        self.device = device
        self.E_nl = E_nl
        self.E_n0 = E_n0
        self.E_max = E_max
        self.l_max = l_max
        self.nu_max = len(E_max) - 1

        self.species_to_species_index = torch.zeros(max(all_species)+1, dtype=torch.long, device=self.device)
        counter = 0
        for species in all_species:
            self.species_to_species_index[species] = counter
            counter +=1

    def forward(self, structure_list):

        composition = self.composition_calculator(structure_list)
        composition = composition.keys_to_samples("species_center")
        composition_energies = composition.block().values.sum(dim=1) * self.multipliers[0]

        radial_spectrum = get_LE_expansion(structure_list, self.radial_spectrum_calculator, self.E_n0, self.E_max[1], self.all_species, rs=True, do_gradients=True, device=self.device)
        radial_spectrum = radial_spectrum.keys_to_samples("a_i") 
        radial_spectrum_species = radial_spectrum.block().samples.column("a_i")
        radial_energies = torch.sum(
            radial_spectrum.block().values * self.multipliers[1][self.species_to_species_index[radial_spectrum_species]],
            dim=1
        )
        
        spherical_expansion = get_LE_expansion(structure_list, self.spherical_expansion_calculator, self.E_nl, self.E_max[2], self.all_species, device=self.device)
        spherical_expansion = spherical_expansion.keys_to_samples("a_i")
        spherical_expansion_species = self.species_to_species_index[spherical_expansion.block({"lam": 0}).samples.column("a_i")]
        spherical_expansion = torch.concatenate(
            [spherical_expansion.block({"lam": l}).values.reshape(spherical_expansion.block({"lam": l}).values.shape[0], -1) for l in range(self.l_max+1)],
            dim=1
        )
        high_order_energies = []
        for nu in range(2, self.nu_max+1):
            """if nu == 3: 
                print(self.combine_indices[nu])
                exit()"""
            high_order_energies.append(
                polyeval(spherical_expansion, self.combine_indices[nu], self.multipliers[nu], spherical_expansion_species)
            )
        energies = torch.stack(
            [composition_energies] + [radial_energies] + high_order_energies,
            dim=1
        )
        total_energy = torch.sum(energies)
        return total_energy

