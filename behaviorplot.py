import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable


def plot_trajectories_and_attacks(ax, data, agent_0_id, agent_0_color, agent_0_label, teammate_color):
    """
    A helper function to plot trajectories for one run.
    - Highlights Agent 0 with a clear trajectory line AND time-colored points.
    - Shows teammates as faint background traces.
    - Marks Agent 0's attacks at their precise location, sized by distance.
    """

    # --- 1. Extract Agent 0 and Teammates' data ---
    agent_0_data = data[data['agent_id'] == agent_0_id]
    teammates_data = data[data['agent_id'] != agent_0_id]

    # --- 2. Plot Teammate Trajectories (as faint background) ---
    df_teammates = pd.DataFrame(teammates_data)
    for _, group in df_teammates.groupby('agent_id'):
        ax.plot(group['x'], group['y'], color=teammate_color, alpha=0.2, linewidth=1, zorder=1)

    # --- 3. Plot Agent 0's Trajectory ---
    # Step 3.1: Draw a solid, semi-transparent background line for clarity
    ax.plot(agent_0_data['x'], agent_0_data['y'],
            color=agent_0_color,
            alpha=0.4,  # Make it semi-transparent
            linewidth=3,  # Make it thick
            zorder=2,
            label=f'{agent_0_label} Trajectory Path')  # For legend

    # Step 3.2: Overlay time-colored scatter points on the line
    # This provides the time information without sacrificing line clarity.

    # Create the truncated colormap for better visibility
    original_cmap = plt.get_cmap(f'{agent_0_color.capitalize()}s')
    start_point = 0.25
    new_colors = original_cmap(np.linspace(start_point, 1.0, 256))
    cmap = LinearSegmentedColormap.from_list("truncated_cmap", new_colors)
    norm = Normalize(vmin=data['step'].min(), vmax=data['step'].max())

    # The scatter points now represent the "flow" of time along the path
    traj_scatter = ax.scatter(agent_0_data['x'], agent_0_data['y'],
                              s=25,  # Slightly larger points
                              c=agent_0_data['step'],
                              cmap=cmap,
                              norm=norm,
                              zorder=3,  # Ensure they are on top of the line
                              edgecolor='none')  # No edge for a smoother look

    # --- 4. Plot Agent 0's Attack Markers ---
    agent_0_attacks = agent_0_data[agent_0_data['is_attacking']]
    marker_style = 'X' if agent_0_label == 'Impulsive' else 'o'

    if len(agent_0_attacks) > 0:
        ax.scatter(agent_0_attacks['x'], agent_0_attacks['y'],
                   s=(agent_0_attacks['attack_dist'] + 1) * 60,  # Make attack markers even larger to stand out
                   c=agent_0_attacks['step'],
                   cmap=cmap,
                   norm=norm,
                   marker=marker_style,
                   edgecolor='white',
                   linewidth=1.5,
                   zorder=4,  # Highest layer
                   label=f'{agent_0_label} Attack')

    return ScalarMappable(norm=norm, cmap=cmap)


# --- Main Script ---
# 1. Load BOTH datasets
# !!! IMPORTANT: Replace with the correct paths to YOUR generated files !!!
impulsive_file = 'behavior_data/full_team_behavior_env_1c3s5z_seed_11_type_normal.npy'
normal_file = 'behavior_data/full_team_behavior_env_1c3s5z_seed_98_type_normal.npy'

if not os.path.exists(impulsive_file) or not os.path.exists(normal_file):
    print(f"Error: Behavioral data files not found!")
    exit()

impulsive_run_data = np.load(impulsive_file)
normal_run_data = np.load(normal_file)

# 2. Create a two-panel figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
fig.suptitle('Behavioral Analysis: Normal vs. Impulsive Agent 0', fontsize=22, fontweight='bold')

# 3. Plot the "Normal" run on the left subplot
mappable_normal = plot_trajectories_and_attacks(ax1, normal_run_data,
                                                agent_0_id=0,
                                                agent_0_color='blue',
                                                agent_0_label='Normal',
                                                teammate_color='gray')
ax1.set_title('Normal Run: Coordinated Movement', fontsize=18)
ax1.set_xlabel('Map X-Coordinate', fontsize=14)
ax1.set_ylabel('Map Y-Coordinate', fontsize=14)
ax1.set_aspect('equal', adjustable='box')
ax1.tick_params(axis='both', which='major', labelsize=12)

# 4. Plot the "Impulsive" run on the right subplot
mappable_impulsive = plot_trajectories_and_attacks(ax2, impulsive_run_data,
                                                   agent_0_id=0,
                                                   agent_0_color='red',
                                                   agent_0_label='Impulsive',
                                                   teammate_color='gray')
ax2.set_title('Impulsive Run: Team Desynchronization', fontsize=18)
ax2.set_xlabel('Map X-Coordinate', fontsize=14)
ax2.set_aspect('equal', adjustable='box')
ax2.tick_params(axis='both', which='major', labelsize=12)

# 5. Create custom legends
legend_elements = [
    Line2D([0], [0], color='blue', lw=3, alpha=0.6, label='Normal Agent 0 Path'),
    Line2D([0], [0], color='red', lw=3, alpha=0.6, label='Impulsive Agent 0 Path'),
    Line2D([0], [0], color='gray', lw=2, alpha=0.4, label='Teammate Paths'),
    Line2D([0], [0], marker='o', color='w', label='Normal Attack (Size~Dist)',
           markerfacecolor='lightblue', markeredgecolor='k', markersize=12),
    Line2D([0], [0], marker='X', color='w', label='Impulsive Attack (Size~Dist)',
           markerfacecolor='lightcoral', markeredgecolor='k', markersize=12)
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=14)

# 6. Add colorbars
cbar_normal = fig.colorbar(mappable_normal, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
cbar_normal.set_label('Timestep (Color darkens over time)', fontsize=12)
cbar_impulsive = fig.colorbar(mappable_impulsive, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
cbar_impulsive.set_label('Timestep (Color darkens over time)', fontsize=12)

# 7. Final adjustments and save
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('behavioral_trajectory_comparison_final.png', dpi=300)
plt.show()