import numpy as np
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


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


def plot_heatmap(states, len_segment, num_interp=200, cmap="magma"):
    x, y = interpolated_polygonal_chains(states, len_segment, num_interp)
    f, ax = plt.subplots()
    ax.set_aspect("equal")
    ax = sns.kdeplot(
        x.ravel(order="F"),
        y.ravel(order="F"),
        cmap=cmap,
        shade=True,
        shade_lowest=False,
        )
    return f, ax