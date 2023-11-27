"""Convert the Espaloma 0.3.0 datasets into a format compatible with the descent"""
import json
import pathlib
import typing

import dgl
import openff.toolkit
import openff.units
import openmm.unit
import torch
from tqdm import tqdm

import descent.targets.energy

HARTEE_TO_KCAL = (
    1.0 * openmm.unit.hartree * openmm.unit.AVOGADRO_CONSTANT_NA
).value_in_unit(openmm.unit.kilocalorie_per_mole)

BOHR_TO_ANGSTROM = (1.0 * openmm.unit.bohr).value_in_unit(openmm.unit.angstrom)


def process_entry(root_dir: pathlib.Path) -> dict[str, typing.Any]:
    mol_dict = json.loads(json.loads((root_dir / "mol.json").read_text()))
    mol_dict["hierarchy_schemes"] = {}
    mol_dict["partial_charge_unit"] = mol_dict["partial_charges_unit"]
    del mol_dict["partial_charges_unit"]
    mol = openff.toolkit.Molecule.from_dict(mol_dict)

    graphs, extra = dgl.load_graphs(str(root_dir / "heterograph.bin"))
    assert len(graphs) == 1
    assert len(extra) == 0

    graph = graphs[0]

    energies = graph.ndata["u_qm"]["g"].flatten() * HARTEE_TO_KCAL

    forces = graph.ndata["u_qm_prime"]["n1"] * (HARTEE_TO_KCAL / BOHR_TO_ANGSTROM)
    forces = torch.swapaxes(forces, 0, 1)

    coords = graph.ndata["xyz"]["n1"] * BOHR_TO_ANGSTROM
    coords = torch.swapaxes(coords, 0, 1)

    return {
        "smiles": mol.to_smiles(mapped=True, isomeric=True),
        "coords": coords.flatten().tolist(),
        "energy": energies.flatten().tolist(),
        "forces": forces.flatten().tolist(),
    }


def main():
    root_dir = pathlib.Path("outputs/8150601")
    output_dir = pathlib.Path("outputs/data-raw")

    smiles_per_set = {}

    for source in ["gen2-opt", "gen2-torsion", "spice-des-monomers", "spice-pubchem"]:
        source_dir = root_dir / source

        entries = [
            f for f in source_dir.glob("*") if f.is_dir() and not f.name.startswith(".")
        ]

        duplicate_dir = root_dir / "duplicated-isomeric-smiles-merge"

        entries_duplicate = list(
            duplicate_dir.glob(f"*/{source.replace('-opt', '')}/*")
        )
        entries_duplicate = [
            f for f in entries_duplicate if f.is_dir() and not f.name.startswith(".")
        ]
        entries.extend(entries_duplicate)

        print(
            f"processing {len(entries)} entries from {source} "
            f"({len(entries_duplicate)} from duplicates)"
        )

        dataset = descent.targets.energy.create_dataset(
            [process_entry(entry) for entry in tqdm(entries)]
        )
        dataset.save_to_disk(output_dir / source)

        unique_smiles = dataset.unique("smiles")
        tqdm.write(
            f"Found {len(dataset)} ({len(unique_smiles)} unique) SMILES in {source}"
        )

        smiles_per_set[source] = dataset.unique("smiles")

    with open(output_dir / "smiles.json", "w") as file:
        json.dump(smiles_per_set, file)


if __name__ == "__main__":
    main()
