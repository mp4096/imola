import numpy as np
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
from .dfd_bindings.rust_bindings import dfd


def _lin_interpolate_1d_nodes(x_coarse, num_interp):
    u_coarse = np.arange(len(x_coarse))
    u_fine = np.linspace(0, len(x_coarse) - 1, num_interp)
    tck = splrep(u_coarse, x_coarse, k=1)
    return splev(u_fine, tck)


def _get_nodes_x(state, len_segment):
    cum_angles = np.cumsum(state[2:])
    nodes_x = np.empty((len(state) - 1,))
    nodes_x[0] = state[0]  # set the x-coordinate
    nodes_x[1:] = np.cos(cum_angles)*len_segment
    nodes_x = np.cumsum(nodes_x)
    return nodes_x


def _get_nodes_y(state, len_segment):
    cum_angles = np.cumsum(state[2:])
    nodes_y = np.empty((len(state) - 1,))
    nodes_y[0] = state[1]  # set the y-coordinate
    nodes_y[1:] = np.sin(cum_angles)*len_segment
    nodes_y = np.cumsum(nodes_y)
    return nodes_y


def get_nodes(state, len_segment):
    return _get_nodes_x(state, len_segment), _get_nodes_y(state, len_segment)


def interpolated_polygonal_chains(states, len_segment, num_interp):
    nodes_x = np.apply_along_axis(
        _get_nodes_x,
        0,
        states,
        len_segment,
        )
    nodes_fine_x = np.apply_along_axis(
        _lin_interpolate_1d_nodes,
        0,
        nodes_x,
        num_interp,
        )

    nodes_y = np.apply_along_axis(
        _get_nodes_y,
        0,
        states,
        len_segment,
        )
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


def compare_to_ground_truth(state, ground_truth, len_segment,
                            camera, num_interp=200):
    state, ground_truth = np.array(state), np.array(ground_truth)
    x, y = interpolated_polygonal_chains(
        state[:, np.newaxis],
        len_segment,
        num_interp,
        )
    model = np.vstack((x.squeeze(), y.squeeze()))
    model = camera.visible_points(model)

    if model.shape[1] == 0 or ground_truth.shape[1] == 0:
        return np.inf
    else:
        return dfd(model, ground_truth)
