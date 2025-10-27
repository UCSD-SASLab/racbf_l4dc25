import matplotlib.pyplot as plt

import numpy as np
import hj_reachability as hj
from torch2jax import t2j, j2t
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# HJR State Space Range
x_range = [-1.9, 1.9]
theta_range = [-np.pi, np.pi]
xdot_range = [-10, 10]
thetadot_range = [-10, 10]

grid_resolution = (51, 51, 51, 51)

# matplotlib settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 14
# thicker lines
plt.rcParams['lines.linewidth'] = 6

state_domain = hj.sets.Box(np.array([x_range[0], theta_range[0], xdot_range[0], thetadot_range[0]]), 
                               np.array([x_range[1], theta_range[1], xdot_range[1], thetadot_range[1]]))

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, 
                                                                grid_resolution, 
                                                                periodic_dims=1)

input_data_raw = np.load("../custom_envs/rl_safeCartpoleRA_hjr_values.npy")

input_datalasttheta = (input_data_raw[:, :, -1:] + input_data_raw[:, :, :1]) / 2
input_data = np.concatenate((input_data_raw, input_datalasttheta), axis=2)

xdot_idx = 25
thetadot_idx = 25

import torch 
def cartpole_sdf(state, 
                 unsafe_x_min, unsafe_x_max, unsafe_vel_max, unsafe_theta_min, unsafe_theta_max, unsafe_thetadot_max, 
                 unsafe_theta_in_range): 

    # unsafe_theta_in_range: if True then the theta range describes what is unsafe
    if isinstance(state, torch.Tensor):
        state = state 
    else: 
        state = torch.tensor(state)

    use_unsafe_theta = True 
    if unsafe_theta_min == unsafe_theta_max: 
        use_unsafe_theta = False 

    x = state[..., 0]
    theta = state[..., 1]
    xdot = state[..., 2]
    thetadot = state[..., 3]

    # Unsafe x: in range is safe 
    unsafe_x = torch.zeros(x.shape).to(state.device)
    greater_than_x = torch.where(x > unsafe_x_max)
    less_than_x = torch.where(x < unsafe_x_min)
    in_range_x = torch.where((unsafe_x_min < x) & (x < unsafe_x_max))
    unsafe_x[greater_than_x] = (unsafe_x_max - x)[greater_than_x] # negative unsafe 
    unsafe_x[less_than_x] = (x - unsafe_x_min)[less_than_x] # negative unsafe
    unsafe_x[in_range_x] = torch.min(x - unsafe_x_min, unsafe_x_max - x)[in_range_x]

    # Unsafe velocity: in range is safe 
    unsafe_xdot = torch.zeros(xdot.shape).to(state.device)
    greater_than_xdot = torch.where(xdot > unsafe_vel_max)
    less_than_xdot = torch.where(xdot < -unsafe_vel_max)
    in_range_xdot = torch.where((-unsafe_vel_max < xdot) & (xdot < unsafe_vel_max))
    unsafe_xdot[greater_than_xdot] = (unsafe_vel_max - xdot)[greater_than_xdot] # negative unsafe
    unsafe_xdot[less_than_xdot] = (xdot - (-1 * unsafe_vel_max))[less_than_xdot] # negative unsafe
    unsafe_xdot[in_range_xdot] = torch.min(xdot - (-1 * unsafe_vel_max), unsafe_vel_max - xdot)[in_range_xdot]

    if use_unsafe_theta: 

        unsafe_theta = torch.zeros(theta.shape).to(state.device)
        greater_than_theta = torch.where(theta > unsafe_theta_max) 
        less_than_theta = torch.where(theta < unsafe_theta_min)
        in_range_theta = torch.where((unsafe_theta_min < theta) & (theta < unsafe_theta_max))
        
        if unsafe_theta_in_range: 
            # Unsafe Theta: in range is unsafe
            unsafe_theta[greater_than_theta] = (theta - unsafe_theta_max)[greater_than_theta]
            unsafe_theta[less_than_theta] = (unsafe_theta_min - theta)[less_than_theta]
            unsafe_theta[in_range_theta] = torch.min(unsafe_theta_min - theta, theta - unsafe_theta_max)[in_range_theta] # negative unsafe
        else: 
            # Safe Theta: in range, Unsafe theta: out of range 
            unsafe_theta[greater_than_theta] = (unsafe_theta_max - theta)[greater_than_theta]
            unsafe_theta[less_than_theta] = (theta - unsafe_theta_min)[less_than_theta]
            unsafe_theta[in_range_theta] = torch.min(theta - unsafe_theta_min, unsafe_theta_max - theta)[in_range_theta]

        # TODO: NOTE: might need to change 
        unsafe_vals = torch.min(unsafe_x, torch.min(unsafe_xdot, unsafe_theta))
    else: 
        unsafe_vals = torch.min(unsafe_x, unsafe_xdot)
    
    # Unsafe thetadot: in range is safe 
    unsafe_thetadot = torch.zeros(thetadot.shape).to(state.device)
    greater_than_thetadot = torch.where(thetadot > unsafe_thetadot_max)
    less_than_thetadot = torch.where(thetadot < -unsafe_thetadot_max)
    in_range_thetadot = torch.where((-unsafe_thetadot_max < thetadot) & (thetadot < unsafe_thetadot_max))
    unsafe_thetadot[greater_than_thetadot] = (unsafe_thetadot_max - thetadot)[greater_than_thetadot] # negative unsafe
    unsafe_thetadot[less_than_thetadot] = (thetadot - (-1 * unsafe_thetadot_max))[less_than_thetadot] # negative unsafe
    unsafe_thetadot[in_range_thetadot] = torch.min(thetadot - (-1 * unsafe_thetadot_max), unsafe_thetadot_max - xdot)[in_range_thetadot]
    
    unsafe_vals = torch.min(unsafe_vals, unsafe_thetadot)
    # import pdb; pdb.set_trace()
    return unsafe_vals


def avoid_sdf(state):
    # Avoid sdf function 
    avoid_unsafe_x_min     = -1.5 
    avoid_unsafe_x_max     = 1.5 
    avoid_unsafe_vel_max   = 20 
    
    avoid_unsafe_theta_min = np.pi/8
    avoid_unsafe_theta_max =  np.pi/4
    avoid_unsafe_thetadot_max = 20

    avoid_unsafe_theta_in_range = True # True = specified theta range is unsafe

    return cartpole_sdf(state, avoid_unsafe_x_min, avoid_unsafe_x_max, avoid_unsafe_vel_max, avoid_unsafe_theta_min, avoid_unsafe_theta_max, avoid_unsafe_thetadot_max, avoid_unsafe_theta_in_range)
  
def reach_sdf(state):
    # Reach sdf function 
    reach_unsafe_x_min     = -1.1
    reach_unsafe_x_max     = -0.8
    reach_unsafe_vel_max   = 0.1

    reach_unsafe_theta_min = -np.pi + 0.25
    reach_unsafe_theta_max =  np.pi - 0.25
    reach_unsafe_thetadot_max = 0.25
    reach_unsafe_theta_in_range = True # True = specified theta range is unsafe
    
    return cartpole_sdf(state, reach_unsafe_x_min, reach_unsafe_x_max, reach_unsafe_vel_max, reach_unsafe_theta_min, reach_unsafe_theta_max, reach_unsafe_thetadot_max, reach_unsafe_theta_in_range)


def visualize_safeset_polar(xdot_idx, thetadot_idx):
    """
    Visualize the safe set using polar coordinates.
    """
    # Create polar plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Create coordinate arrays
    x_coords = grid.coordinate_vectors[0]
    theta_coords = grid.coordinate_vectors[1]
    theta_coords_alt = np.concatenate((theta_coords, np.array([3.11])), axis=0)
    # Flip horizontally (left becomes right) and rotate by 90 degrees
    # First flip horizontally, then rotate by 90 degrees (π/2 radians)
    theta_coords_flipped = -theta_coords + np.pi/2
    theta_coords_flipped_alt = -theta_coords_alt + np.pi/2

    avoid_values = t2j(avoid_sdf(j2t(grid.states)))
    reach_values = t2j(reach_sdf(j2t(grid.states)))
    avoid_values = np.concatenate((avoid_values, avoid_values[:, :1]), axis=1)
    reach_values = np.concatenate((reach_values, reach_values[:, :1]), axis=1)

    # Create flipped theta grid for plotting
    theta_grid_flipped_up = np.concatenate((theta_coords_flipped, theta_coords_flipped[:1]), axis=0)

    ax.contourf(theta_grid_flipped_up, x_coords, avoid_values[:, :, xdot_idx, thetadot_idx], levels=[-10, 0], colors='red')
    ax.contourf(theta_grid_flipped_up, x_coords, reach_values[:, :, xdot_idx, thetadot_idx], levels=[0, 5], colors='green')

    # Generate video of the safe set (animation matplotlib)
    for i in range(0, 40, 2):
        ax.contour(theta_coords_flipped_alt, x_coords, input_data[i, :, :, xdot_idx, thetadot_idx], levels=[0], colors='orange')
    
    # Get actual velocity values
    xdot_val = grid.coordinate_vectors[2][xdot_idx]
    thetadot_val = grid.coordinate_vectors[3][thetadot_idx]
    print("xdot_val: ", xdot_val)
    print("thetadot_val: ", thetadot_val)
    
    # Add labels
    ax.set_title(f'Safe Set Visualization in Polar Coordinates\n' + 
                f'Fixed: $\\dot{{x}}$={xdot_val:.1f}, $\\dot{{\\theta}}$={thetadot_val:.1f}', 
                pad=30, fontsize=14)
    
    # Add axis labels for polar plot
    ax.set_xlabel('Angle ($\\theta$)', labelpad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.text(0.3, 0.5, 'Position (x)', ha='center', va='center', fontsize=14, rotation=25)
    
    # Swap tick labels: 0° becomes 90°, 90° becomes 0° (with degree symbols)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], 
                      ['90°', '45°', '0°', '315°', '270°', '225°', '180°', '135°'])
    
    plt.tight_layout()
    return fig, ax


def animate_safeset_polar(xdot_idx, thetadot_idx, save_path='safeset_polar.mp4'):
    """
    Create an animation of the safe set in polar coordinates from -1 to 0.
    """

    # Create polar figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Create coordinate arrays
    x_coords = grid.coordinate_vectors[0]
    theta_coords = grid.coordinate_vectors[1]
    theta_coords_alt = np.concatenate((theta_coords, np.array([3.11])), axis=0)
    # Flip horizontally (left becomes right) and rotate by 90 degrees
    # First flip horizontally, then rotate by 90 degrees (π/2 radians)
    theta_coords_flipped = -theta_coords + np.pi/2
    theta_coords_flipped_alt = -theta_coords_alt + np.pi/2

    avoid_values = t2j(avoid_sdf(j2t(grid.states)))
    reach_values = t2j(reach_sdf(j2t(grid.states)))
    avoid_values = np.concatenate((avoid_values, avoid_values[:, :1]), axis=1)
    reach_values = np.concatenate((reach_values, reach_values[:, :1]), axis=1)

    # Create flipped theta grid for plotting
    theta_grid_flipped_up = np.concatenate((theta_coords_flipped, theta_coords_flipped[:1]), axis=0)

    ax.contourf(theta_grid_flipped_up, x_coords, avoid_values[:, :, xdot_idx, thetadot_idx], levels=[-10, 0], colors='red')
    ax.contourf(theta_grid_flipped_up, x_coords, reach_values[:, :, xdot_idx, thetadot_idx], levels=[0, 5], colors='blue')


    # Animation frame (line that moves)
    contour_plot = [ax.contour(theta_coords_flipped_alt, x_coords, input_data[0, :, :, xdot_idx, thetadot_idx],
                               levels=[0], colors='orange')]
    ax.set_ylim(grid.coordinate_vectors[0][0], grid.coordinate_vectors[0][-1])
    
    def update(frame):
        # Clear old contour
        for coll in contour_plot[0].collections:
            coll.remove()
        # Draw new contour
        contour_plot[0] = ax.contour(theta_coords_flipped_alt, x_coords, input_data[frame, :, :, xdot_idx, thetadot_idx],
                                     levels=[0], colors='green')
        ax.set_title(f"Safe Set Evolution (t={0.1 * frame:.1f}s)")
        return contour_plot

    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], 
                      ['90°', '45°', '0°', '315°', '270°', '225°', '180°', '135°'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=reversed(np.arange(0, 101)), interval=200, blit=False)

    # Save video
    ani.save(save_path, writer=PillowWriter(fps=10), dpi=300)
    plt.close(fig)
    
    print(f"Saved animation to {save_path}")


# Create visualizations
print("Creating polar safe set visualizations...")

# Single slice visualization
fig1, ax1 = visualize_safeset_polar(xdot_idx, thetadot_idx)
plt.savefig('safe_set_polar.png', dpi=300, bbox_inches='tight')
plt.show()

# Create animation of the safe set
animate_safeset_polar(xdot_idx, thetadot_idx, save_path='safeset_polar.gif')

print("Polar visualizations saved!")
