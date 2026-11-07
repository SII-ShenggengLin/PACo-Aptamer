import re
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .data import PACO9, ss_to_pairs

MODIFIED_RESIDUES = {
    "1MA": "A", "MIA": "A", "6MA": "A", "2MA": "A", "M2A": "A", "M6A": "A",
    "M7A": "A", "MMU": "A", "ZAD": "A", "A23": "A", "A2M": "A", "N6G": "A",
    "5MC": "C", "M5C": "C", "CMO": "C", "CCC": "C", "CBR": "C", "C2L": "C",
    "C43": "C", "C5L": "C", "C7M": "C", "CD0": "C", "CFL": "C", "CG1": "C",
    "CH6": "C", "CNO": "C", "CR2": "C", "CRB": "C", "CSL": "C", "TCY": "C",
    "2MG": "G", "7MG": "G", "M2G": "G", "G7M": "G", "MAG": "G", "GDO": "G",
    "G25": "G", "G2L": "G", "G36": "G", "G37": "G", "G49": "G", "GH3": "G",
    "GHT": "G", "GIA": "G", "GMU": "G", "GNG": "G", "GOM": "G", "GSO": "G",
    "GTP": "G", "NMG": "G", "OGU": "G", "QUO": "G", "YYG": "G", "YG": "G",
    "YOF": "G", "O2G": "G", "O3G": "G", "O8G": "G", "PRF": "G", "7GU": "G",
    "PSU": "U", "5MU": "U", "H2U": "U", "OMU": "U", "UR3": "U", "20U": "U",
    "4SU": "U", "M5U": "U", "5OHU": "U", "5BU": "U", "DHU": "U", "S4U": "U",
    "M3U": "U", "MNU": "U", "U2L": "U", "U25": "U", "U31": "U", "U36": "U",
    "UAR": "U", "UBD": "U", "UCL": "U", "UFR": "U", "UFT": "U", "UMA": "U",
    "UMC": "U", "UML": "U", "UMO": "U", "UOX": "U", "UPA": "U", "UPL": "U",
    "UPT": "U", "URD": "U", "URM": "U", "URO": "U", "US1": "U", "US5": "U",
    "USE": "U", "USM": "U", "UTR": "U", "XUA": "U", "ZTH": "U", "I4U": "U",
    "DHE": "U", "H3U": "U", "HBI": "U", "HDU": "U", "HIA": "U", "HMU": "U",
    "HPU": "U", "HTA": "U", "IMC": "U", "KAG": "U", "LHG": "U", "MA6": "U",
    "MAM": "U", "MAP": "U", "MAQ": "U", "MAU": "U", "MBG": "U", "MDO": "U",
    "MED": "U", "MG3": "U", "MGU": "U", "MIS": "U", "MO6": "U", "MOG": "U",
    "MOW": "U", "MPG": "U", "MPN": "U", "MTU": "U", "MVA": "U", "N2G": "U",
    "NC6": "U", "NRQ": "U", "O4U": "U", "O5U": "U", "OA1": "U", "ONH": "U",
    "ONL": "U", "ONP": "U", "ORD": "U", "OTG": "U", "P5U": "U", "PBG": "U",
    "PGP": "U", "PGV": "U", "PHU": "U", "PMP": "U", "PPN": "U", "PR5": "U",
    "PSD": "U", "PTR": "U", "PYO": "U", "R5P": "U", "S2C": "U", "S2G": "U",
    "S6G": "U", "SEL": "U", "SGP": "U", "SHP": "U", "SMP": "U", "SRA": "U",
    "SRS": "U", "SST": "U", "STU": "U", "SUL": "U", "T6A": "U", "TAF": "U",
    "TBR": "U", "TDR": "U", "TFA": "U", "TGP": "U", "TGT": "U", "THX": "U",
    "TLC": "U", "TMD": "U", "TMS": "U", "TRC": "U", "TRQ": "U", "TSP": "U",
    "TTS": "U", "TUR": "U", "TYP": "U", "U8O": "U", "UDP": "U", "UNK": "U",
    "VAS": "U", "WBU": "U", "XCP": "U", "YCM": "U", "ZCT": "U", "ZDU": "U",
    "ZGU": "U", "ZYD": "U", "ZYT": "U", "I": "U",
}
STANDARD_RESIDUES = {"A", "C", "G", "U"}


def read_rna_chain(pdb_path, chain=None):
    residues = OrderedDict()
    available_chains = []
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or line[16] not in (" ", "A"):
                continue
            resname = line[17:20].strip().upper()
            if resname not in STANDARD_RESIDUES and resname not in MODIFIED_RESIDUES:
                continue
            chain_id = line[21].strip()
            if chain_id not in available_chains:
                available_chains.append(chain_id)
            if chain is not None and chain_id != chain:
                continue
            key = (chain_id, line[22:26].strip(), line[26].strip())
            if key not in residues:
                residues[key] = {"resname": resname, "atoms": {}}
            atom = line[12:16].strip().replace("*", "'")
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            residues[key]["atoms"][atom] = xyz
    if not residues:
        raise ValueError(f"No supported RNA residues found in {pdb_path}.")
    selected_chains = list(dict.fromkeys(key[0] for key in residues))
    if chain is None and len(selected_chains) != 1:
        raise ValueError(
            f"The PDB contains RNA chains {selected_chains}; specify one with --chain."
        )
    if chain is not None and chain not in selected_chains:
        raise ValueError(f"RNA chain {chain!r} was not found; available: {available_chains}.")
    if chain is not None:
        residues = OrderedDict((key, value) for key, value in residues.items() if key[0] == chain)

    sequence = "".join(
        MODIFIED_RESIDUES.get(residue["resname"], residue["resname"])
        for residue in residues.values()
    )
    coords = np.full((len(sequence), len(PACO9), 3), np.nan, dtype=np.float32)
    for i, residue in enumerate(residues.values()):
        for j, atom in enumerate(PACO9):
            if atom in residue["atoms"]:
                coords[i, j] = residue["atoms"][atom]
    return sequence, coords


def predict_secondary_structure(sequence, rnafold="RNAfold"):
    executable = shutil.which(rnafold) if not Path(rnafold).is_file() else rnafold
    if executable is None:
        raise FileNotFoundError(
            "RNAfold was not found. Install ViennaRNA or pass --rnafold /path/to/RNAfold."
        )
    result = subprocess.run(
        [executable, "--noPS"],
        input=sequence + "\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"RNAfold failed: {result.stderr.strip()}")
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if fields and len(fields[0]) == len(sequence) and re.fullmatch(
            r"[().\[\]{}<>]+", fields[0]
        ):
            return fields[0]
    raise RuntimeError("Could not parse the RNAfold secondary-structure output.")


def feature_from_pdb(pdb_path, chain=None, rnafold="RNAfold", secondary_structure=None):
    sequence, coords = read_rna_chain(pdb_path, chain)
    if secondary_structure is None:
        secondary_structure = predict_secondary_structure(sequence, rnafold)
    if len(secondary_structure) != len(sequence):
        raise ValueError("The secondary-structure string must match the RNA length.")
    return {
        "coords9": coords,
        "ss_pairs": ss_to_pairs(secondary_structure),
        "seq": sequence,
    }
