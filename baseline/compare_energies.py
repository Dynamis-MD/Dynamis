
import numpy as np
import os
import glob
import sys

def main():
    # 1. Find common steps
    serial_files = glob.glob("results/serial_md/traj_step_*.npz")
    dist_files = glob.glob("results/md/traj_step_*_rank0.npz")
    
    if not serial_files:
        print("Error: No serial files found in results/serial_md/")
        return
    if not dist_files:
        print("Error: No distributed files found in results/md/")
        return
        
    serial_steps = [int(f.split('_')[-1].split('.')[0]) for f in serial_files]
    dist_steps = [int(f.split('_')[-2].split('.')[0]) if 'rank' in f else int(f.split('_')[-1].split('.')[0]) for f in dist_files]
    
    common_steps = sorted(list(set(serial_steps) & set(dist_steps)))
    if not common_steps:
        print(f"Error: No common steps found. Serial: {serial_steps}, Dist: {dist_steps}")
        return
        
    step = common_steps[0]
    print(f"Comparing Step {step}...")
    
    serial_path = f"results/serial_md/traj_step_{step}.npz"
    # Load Serial Data
    print(f"Loading {serial_path}...")
    serial_data = np.load(serial_path)
    serial_energies = serial_data['per_atom_energies']
    serial_gid = serial_data['global_id']
    
    # Sort by gid
    serial_order = np.argsort(serial_gid)
    serial_energies_sorted = serial_energies[serial_order]
    serial_gid_sorted = serial_gid[serial_order]
    
    print(f"Serial Total Energy: {serial_energies.sum():.4f}")

    # 2. Load Distributed Data
    dist_paths = glob.glob(f"results/md/traj_step_{step}_rank*.npz")
    if not dist_paths:
        print("Error: No distributed trajectory files found in results/md/")
        return
        
    print(f"Found {len(dist_paths)} distributed files.")
    
    dist_energies_list = []
    dist_gid_list = []
    
    for p in dist_paths:
        data = np.load(p)
        if 'per_atom_energies' not in data:
            print(f"Warning: 'per_atom_energies' not found in {p}")
            continue
        dist_energies_list.append(data['per_atom_energies'])
        if 'global_id' in data:
            dist_gid_list.append(data['global_id'])
        else:
             print(f"Warning: 'global_id' not found in {p}")

    if not dist_energies_list:
        print("No valid distributed data found.")
        return

    dist_energies = np.concatenate(dist_energies_list)
    dist_gid = np.concatenate(dist_gid_list)
    
    # Sort by gid
    dist_order = np.argsort(dist_gid)
    dist_energies_sorted = dist_energies[dist_order]
    dist_gid_sorted = dist_gid[dist_order]
    
    print(f"Distributed Total Energy: {dist_energies.sum():.4f}")
    
    # 3. Compare
    if len(serial_energies_sorted) != len(dist_energies_sorted):
        print(f"Error: Atom count mismatch! Serial: {len(serial_energies_sorted)}, Dist: {len(dist_energies_sorted)}")
        # Try to align intersection
        common_gid = np.intersect1d(serial_gid_sorted, dist_gid_sorted)
        print(f"Intersecting on {len(common_gid)} atoms...")
        
        # Filter
        serial_mask = np.isin(serial_gid_sorted, common_gid)
        dist_mask = np.isin(dist_gid_sorted, common_gid)
        
        serial_energies_sorted = serial_energies_sorted[serial_mask]
        dist_energies_sorted = dist_energies_sorted[dist_mask]
        
    diff = dist_energies_sorted - serial_energies_sorted
    abs_diff = np.abs(diff)
    
    print("-" * 40)
    print(f"Comparison Statistics:")
    print(f"Max Diff: {np.max(abs_diff):.6f} eV")
    print(f"Mean Diff: {np.mean(abs_diff):.6f} eV")
    print(f"RMSE: {np.sqrt(np.mean(diff**2)):.6f} eV")
    print(f"Sum Diff: {np.sum(diff):.6f} eV")
    print("-" * 40)
    
    # Top 10 discrepancies
    top_indices = np.argsort(abs_diff)[-20:]
    print("Top 20 Discrepancies:")
    print("GID | Serial | Dist | Diff")
    for idx in top_indices:
        gid = serial_gid_sorted[idx] # Assuming aligned
        s_val = serial_energies_sorted[idx]
        d_val = dist_energies_sorted[idx]
        d = diff[idx]
        print(f"{gid:4d} | {s_val:.4f} | {d_val:.4f} | {d:.4f}")

if __name__ == "__main__":
    main()
