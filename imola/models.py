import numpy as np
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
from .dfd_bindings.rust_bindings import dfd


def _lin_interpolate_1d_nodes(x_coarse, num_interp):
    u_coarse = np.arange(len(x_coarse))
    u_fine = np.linspace(0, len(x_coarse) - 1, num_interp)
    tck = splrep(u_coarse, x_coarse, k=1)
    return splev(u_fine, tck)


def interpolated_polygonal_chains(states, len_segment, num_interp):
    num_chains = states.shape[1]
    num_nodes = states.shape[0] - 1

    cum_angles = np.cumsum(states[2:, :], axis=0)

    nodes_x = np.empty((num_nodes, num_chains))
    nodes_x[0, :] = states[0, :]  # set the x-coordinate of all first nodes
    nodes_x[1:, :] = np.cos(cum_angles)*len_segment
    nodes_x = np.cumsum(nodes_x, axis=0)
    nodes_fine_x = np.apply_along_axis(
        _lin_interpolate_1d_nodes,
        0,
        nodes_x,
        num_interp,
        )

    nodes_y = np.empty((num_nodes, num_chains))
    nodes_y[0, :] = states[1, :]  # set the y-coordinate of all first nodes
    nodes_y[1:, :] = np.sin(cum_angles)*len_segment
    nodes_y = np.cumsum(nodes_y, axis=0)
    nodes_fine_y = np.apply_along_axis(
        _lin_interpolate_1d_nodes,
        0,
        nodes_y,
        num_interp,
        )

    return nodes_fine_x, nodes_fine_y


def plot_heatmap(states, len_segment, num_interp=200,
                 gridsize=50, cmap="inferno"):
    x, y = interpolated_polygonal_chains(states, len_segment, num_interp)
    f, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.hexbin(
        x.ravel(order="F"),
        y.ravel(order="F"),
        gridsize=gridsize,
        cmap=cmap)
    return f, ax


def compare_to_ground_truth(state, ground_truth, len_segment, num_interp=200):
    state, ground_truth = np.array(state), np.array(ground_truth)
    x, y = interpolated_polygonal_chains(
        state[:, np.newaxis],
        len_segment,
        num_interp,
        )
    model = np.vstack((x.squeeze(), y.squeeze()))
    return dfd(model, ground_truth)
