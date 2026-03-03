
import numpy as np
import os
import glob
import sys

def main():
    # 1. Find common steps
    serial_files = glob.glob("results/uma_4k_4GPU/traj_step_*.npz")
    dist_files = glob.glob("results/dist_4k_4GPU/traj_step_*_rank0.npz")
    
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
    
    serial_path = f"results/uma_4k_4GPU/traj_step_{step}.npz"
    # Load Serial Data
    print(f"Loading {serial_path}...")
    serial_data = np.load(serial_path)
    serial_forces = serial_data['forces']
    serial_pos = serial_data['positions']
    serial_gid = serial_data['global_id']
    
    # Sort by gid
    serial_order = np.argsort(serial_gid)
    serial_forces = serial_forces[serial_order]
    serial_pos = serial_pos[serial_order]
    serial_gid = serial_gid[serial_order]

    # 2. Load Distributed Data
    dist_paths = glob.glob(f"results/dist_4k_4GPU/traj_step_{step}_rank*.npz")
    dist_forces_list = []
    dist_pos_list = []
    dist_gid_list = []
    
    for p in dist_paths:
        data = np.load(p)
        dist_forces_list.append(data['forces'])
        dist_pos_list.append(data['positions'])
        if 'global_id' in data:
            dist_gid_list.append(data['global_id'])
            
    dist_forces = np.concatenate(dist_forces_list)
    dist_pos = np.concatenate(dist_pos_list)
    dist_gid = np.concatenate(dist_gid_list)
    
    # Sort by gid
    dist_order = np.argsort(dist_gid)
    dist_forces = dist_forces[dist_order]
    dist_pos = dist_pos[dist_order]
    dist_gid = dist_gid[dist_order]
    
    # 3. Compare
    if len(serial_forces) != len(dist_forces):
        print(f"Error: Atom count mismatch! Serial: {len(serial_forces)}, Dist: {len(dist_forces)}")
        # Filter
        common_gid = np.intersect1d(serial_gid, dist_gid)
        serial_mask = np.isin(serial_gid, common_gid)
        dist_mask = np.isin(dist_gid, common_gid)
        
        serial_forces = serial_forces[serial_mask]
        dist_forces = dist_forces[dist_mask]
        serial_pos = serial_pos[serial_mask]
        dist_pos = dist_pos[dist_mask]
        serial_gid = serial_gid[serial_mask]
        
    # Position Diff
    pos_diff = dist_pos - serial_pos
    pos_diff_norm = np.linalg.norm(pos_diff, axis=1)
    print(f"Max Position Diff: {np.max(pos_diff_norm):.6f}")
    
    # Force Diff
    force_diff = dist_forces - serial_forces
    force_diff_norm = np.linalg.norm(force_diff, axis=1)
    
    print("-" * 40)
    print(f"Force Comparison Statistics:")
    print(f"Max Diff Norm: {np.max(force_diff_norm):.6f}")
    print(f"Mean Diff Norm: {np.mean(force_diff_norm):.6f}")
    print("-" * 40)
    
    # Top 20 discrepancies
    top_indices = np.argsort(force_diff_norm)[-20:]
    print("Top 20 Force Discrepancies:")
    print("GID | F_serial_norm | F_dist_norm | Diff_norm | Pos_diff")
    for idx in top_indices:
        gid = serial_gid[idx]
        s_norm = np.linalg.norm(serial_forces[idx])
        d_norm = np.linalg.norm(dist_forces[idx])
        diff = force_diff_norm[idx]
        p_diff = pos_diff_norm[idx]
        print(f"{gid:4d} | {s_norm:.4f} | {d_norm:.4f} | {diff:.4f} | {p_diff:.6f}")

if __name__ == "__main__":
    main()
