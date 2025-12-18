import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot_3d_trajectory(ax, data, agent_0_id, agent_0_color, agent_0_label, teammate_color):
    """
    Plots the 3D trajectory (x, y, time) for Agent 0 and its teammates' 2D projection.
    """
    # --- 1. Extract Data ---
    agent_0_data = data[data['agent_id'] == agent_0_id]
    teammates_data = data[data['agent_id'] != agent_0_id]

    # --- 2. Plot Teammates' Trajectories on the "Floor" (z=0) ---
    df_teammates = pd.DataFrame(teammates_data)
    for _, group in df_teammates.groupby('agent_id'):
        ax.plot(group['x'], group['y'], zs=0, zdir='z', color=teammate_color, alpha=0.2, linewidth=1.5)

    # Add a single proxy line for the legend
    ax.plot([], [], [], color=teammate_color, alpha=0.4, linewidth=2, label='Teammate Paths (on floor)')

    # --- 3. Plot Agent 0's 3D Trajectory ---
    x_path = agent_0_data['x']
    y_path = agent_0_data['y']
    z_path = agent_0_data['step'] # Time is the Z-axis
    ax.plot(x_path, y_path, z_path, color=agent_0_color, linewidth=2.5, alpha=0.8, label=f'{agent_0_label} Agent 0 Trajectory')

    # --- 4. Mark Initial and End Positions ---
    # Initial Position
    ax.scatter(x_path[0], y_path[0], z_path[0], c=agent_0_color, marker='o', s=100, edgecolor='black', depthshade=True, label=f'{agent_0_label} Initial Position')
    # End Position
    ax.scatter(x_path[-1], y_path[-1], z_path[-1], c=agent_0_color, marker='*', s=200, edgecolor='black', depthshade=True, label=f'{agent_0_label} End Position')

    # --- 5. Visualize Attacks as Vertical "Energy" Bars/Disks ---
    agent_0_attacks = agent_0_data[agent_0_data['is_attacking']]
    if len(agent_0_attacks) > 0:
        for attack in agent_0_attacks:
            x, y, z = attack['x'], attack['y'], attack['step']
            # Draw a vertical line to represent the attack event
            # The length of the line can represent attack distance, but this can be messy.
            # A simpler, clearer approach is a prominent marker.
            ax.plot([x, x], [y, y], [z-1, z+1], color=agent_0_color, linewidth=5, solid_capstyle='round', alpha=0.9)
            # Add a scatter point on top to make it more visible
            ax.scatter(x, y, z, s=(attack['attack_dist']+1)*20, c=agent_0_color, marker='h', edgecolor='white', linewidth=1.5, depthshade=True)

# --- Main Script ---
# 1. Load Data
impulsive_file = 'behavior_data/full_team_behavior_env_1c3s5z_seed_11_type_normal.npy'
normal_file = 'behavior_data/full_team_behavior_env_1c3s5z_seed_98_type_normal.npy'

if not os.path.exists(impulsive_file) or not os.path.exists(normal_file):
    print("Error: Behavioral data files not found!")
    exit()

impulsive_run_data = np.load(impulsive_file)
normal_run_data = np.load(normal_file)

# 2. Create Figure
fig = plt.figure(figsize=(20, 10))
fig.suptitle('3D Behavioral Analysis: Low Impulsive vs. Impulsive Agent 0', fontsize=22, fontweight='bold')

# 3. Create Left Subplot (Normal Run)
ax1 = fig.add_subplot(121, projection='3d')
plot_3d_trajectory(ax1, normal_run_data, 0, 'blue', 'Low Impulsive', 'gray')
ax1.set_title('Low Impulsive Run: Coordinated & Sustained Trajectory', fontsize=16)
ax1.set_xlabel('Map X-Coordinate [m]')
ax1.set_ylabel('Map Y-Coordinate [m]')
ax1.set_zlabel('Timestep (Survival Time)')
ax1.legend()
ax1.view_init(elev=30, azim=-60) # Set viewing angle

# 4. Create Right Subplot (Impulsive Run)
ax2 = fig.add_subplot(122, projection='3d')
plot_3d_trajectory(ax2, impulsive_run_data, 0, 'red', 'Impulsive', 'gray')
ax2.set_title('Impulsive Run: Desynchronized & Short-lived Trajectory', fontsize=16)
ax2.set_xlabel('Map X-Coordinate [m]')
ax2.set_ylabel('Map Y-Coordinate [m]')
ax2.set_zlabel('Timestep (Survival Time)')
ax2.legend()
ax2.view_init(elev=30, azim=-60) # Use the same angle for direct comparison

# 5. Save and Show
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('behavioral_trajectory_3D_comparison.png', dpi=900)
plt.show()