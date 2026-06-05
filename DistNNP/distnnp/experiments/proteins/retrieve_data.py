import urllib.request
import gzip

# Hydrogen atoms are not present in the 3J3Q cryo-EM structure and are not added here.
# This simulation uses heavy atoms only. Proper hydrogen placement could be required
# when swapping in a real NNP model.

def download_and_convert_to_xyz(pdb_id="3J3Q"):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    cif_filename = f"{pdb_id}.cif.gz"
    xyz_filename = f"{pdb_id}.xyz"

    print(f"Downloading {pdb_id} from {url}...")
    # This file is around 40 MB compressed
    urllib.request.urlretrieve(url, cif_filename)
    print("Download complete.")

    print("Parsing mmCIF file and extracting atoms (this may take a minute)...")
    
    # Store lines in memory to write them all at once at the end
    # (~70MB in RAM for 2.4 million atoms)
    atom_lines = []
    
    # Use python's gzip to open and read the compressed text file natively
    with gzip.open(cif_filename, "rt") as f:
        in_atom_site = False
        columns = []
        sym_idx, x_idx, y_idx, z_idx = -1, -1, -1, -1

        for line in f:
            line = line.strip()

            if line.startswith("loop_"):
                # Reset column definitions when a new loop block starts
                columns = []
                in_atom_site = False
            elif line.startswith("_atom_site."):
                in_atom_site = True
                columns.append(line)
            elif in_atom_site and (line.startswith("ATOM") or line.startswith("HETATM")):
                if sym_idx == -1:
                    # Dynamically identify the indices of the required columns
                    sym_idx = columns.index("_atom_site.type_symbol")
                    x_idx = columns.index("_atom_site.Cartn_x")
                    y_idx = columns.index("_atom_site.Cartn_y")
                    z_idx = columns.index("_atom_site.Cartn_z")

                parts = line.split()
                # Extract Element and X,Y,Z coordinates
                element = parts[sym_idx]
                x = parts[x_idx]
                y = parts[y_idx]
                z = parts[z_idx]

                atom_lines.append(f"{element} {x} {y} {z}\n")

    print(f"Successfully extracted {len(atom_lines)} atoms.")
    print(f"Writing structure to {xyz_filename}...")
    
    with open(xyz_filename, "w") as out_f:
        # Standard .xyz formatting:
        # Line 1: Atom count
        # Line 2: Comment line
        # Line 3+: Element X Y Z
        out_f.write(f"{len(atom_lines)}\n")
        out_f.write(f"HIV-1 capsid {pdb_id}\n")
        out_f.writelines(atom_lines)
        
    print("Conversion finished!")

if __name__ == "__main__":
    download_and_convert_to_xyz("3J3Q")