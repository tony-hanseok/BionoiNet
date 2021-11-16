import argparse
import pandas as pd
import numpy as np

COLUMN_NAMES = (
    "atom_id",
    "atom_name",
    "x",
    "y",
    "z",
    "atom_type",
    "subst_id",
    "subst_name",
    "charge",
)


def get_colorgen(colorby):
    # if colorby == "atom_type":
    #     color_map = "./cmaps/atom_cmap.csv"
    # else:
    #     color_map = "./cmaps/res_hydro_cmap.csv"

    if colorby == "atom_type":
        color_map = "./bionoi/cmaps/atom_cmap.csv"
    else:
        color_map = "./bionoi/cmaps/res_hydro_cmap.csv"

    # Check for color mapping file, make dict
    with open(color_map, "rt") as color_mapF:
        # Parse color map file
        color_map = np.array(
            [
                line.replace("\n", "").split(";")
                for line in color_mapF.readlines()
                if not line.startswith("#")
            ]
        )
        # Create color dictionary
        color_map = {
            code: {"color": color, "definition": definition}
            for code, definition, color in color_map
        }
        return color_map


def parse_molecule(lines):
    pdb_fname = lines[0].strip()
    num_atoms, num_bonds, *_ = lines[1].strip().split()
    mol_type = lines[2].strip()
    charge_type = lines[3].strip()
    return {
        "pdb_fname": pdb_fname,
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
        "mol_type": mol_type,
        "charge_type": charge_type,
    }


def parse_atom(lines):
    atom_df = pd.DataFrame(data=[line.split() for line in lines], columns=COLUMN_NAMES)
    color_map = get_colorgen(colorby="residue_type")

    valid_atom_ids = []
    invalid_atom_ids = []
    for idx, row in atom_df.iterrows():
        if row["subst_name"][:3] in color_map.keys():
            valid_atom_ids.append(idx)
        else:
            invalid_atom_ids.append(idx)
    return atom_df, valid_atom_ids, invalid_atom_ids


def map_new_atom_ids(atom_df, valid_atom_ids, invalid_atom_ids):
    new_atom_df = pd.concat(
        [atom_df.iloc[valid_atom_ids], atom_df.iloc[invalid_atom_ids]]
    )
    new_atom_df.index = [i + 1 for i in range(len(new_atom_df))]
    new_atom_ids = {k: v for k, v in zip(new_atom_df["atom_id"], new_atom_df.index)}
    return new_atom_df[:len(valid_atom_ids)], new_atom_ids


def parse_bond(lines, invalid_atom_ids, new_atom_ids):
    valid_bonds = []
    for line in lines:
        _, origin_atom, to_atom, bond_type = line.split()

        if origin_atom in invalid_atom_ids or to_atom in invalid_atom_ids:
            continue
        valid_bonds.append(
            [
                str(len(valid_bonds) + 1),
                str(new_atom_ids[origin_atom]),
                str(new_atom_ids[to_atom]),
                bond_type,
            ]
        )
    return valid_bonds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mol",
        default="/home/tony/sandbox/pdb/mol2/3D2R.mol2",
        required=False,
        help="the protein/ligand mol2 file",
    )
    args = parser.parse_args()

    molecule_lines = []
    atom_lines = []
    bond_lines = []
    with open(args.mol, "r") as f:
        for line in f:
            line = line.strip()
            if line == "@<TRIPOS>MOLECULE":
                lines = molecule_lines
            elif line == "@<TRIPOS>ATOM":
                lines = atom_lines
            elif line == "@<TRIPOS>BOND":
                lines = bond_lines
            else:
                lines.append(line)

    parsed_molecule = parse_molecule(lines=molecule_lines)
    atom_df, valid_atom_ids, invalid_atom_ids = parse_atom(lines=atom_lines)
    new_atom_df, new_atom_ids = map_new_atom_ids(
        atom_df=atom_df,
        valid_atom_ids=valid_atom_ids,
        invalid_atom_ids=invalid_atom_ids,
    )
    new_atom_df.set_index("atom_id")
    parsed_bonds = parse_bond(
        lines=bond_lines, invalid_atom_ids=invalid_atom_ids, new_atom_ids=new_atom_ids
    )

    new_molecule_lines = [
        parsed_molecule["pdb_fname"],
        f" {len(new_atom_df)} {len(parsed_bonds)} 0 0 0",
        parsed_molecule["mol_type"],
        parsed_molecule["charge_type"],
    ]

    new_atom_lines = ["\t".join(row) for _, row in new_atom_df.iterrows()]
    new_bond_lines = [" ".join(bond_info) for bond_info in parsed_bonds]

    with open(f"{args.mol}.processed", "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write("\n".join(new_molecule_lines))
        f.write("\n\n")
        f.write("@<TRIPOS>ATOM\n")
        f.write("\n".join(new_atom_lines))
        f.write("\n")
        f.write("@<TRIPOS>BOND\n")
        f.write("\n".join(new_bond_lines))
