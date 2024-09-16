import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from kitman.visualizations import draw_soccer_field
from matplotlib.collections import PatchCollection
from skimage import io


def sample_colors_from_cmap(N, cmap_name='hsv'):
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, N))
    return colors


def plot_players_overlaid(data, player_colors=None, figsize=(15, 7.5), title=None, ax=None, fontsize=14):
    frame = io.imread(data['frame_path'])

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    if title:
        ax.set_title(title, fontweight="bold", fontsize=fontsize)

    ax.imshow(frame)

    categories = np.unique(data.player_categories)

    if not player_colors:
        num_player_categories = len(categories)
        player_colors = {cat: col for cat, col in zip(categories, sample_colors_from_cmap(num_player_categories))}

    patches = {c: [] for c in data.player_categories}
    for c, p in zip(data.player_categories, data.patches):
        patches[c].append(p)

    for i in patches.keys():
        p = PatchCollection(patches[i], color=player_colors[i], alpha=0.4)
        ax.add_collection(p)

    if 'fig' in locals():
        fig.tight_layout()
        return fig, ax


def plot_players_projection(data, player_colors=None, node_size=10, figsize=(15, 7.5), title=None, ax=None):
    if not ax:
        fig, ax = draw_soccer_field(figsize=figsize)
    else:
        draw_soccer_field(figsize=figsize, ax=ax)

    if title:
        ax.set_title(title, fontweight="bold", fontsize=14)

    if not player_colors:
        categories = np.unique(data.player_categories)
        num_player_categories = len(categories)
        player_colors = {cat: col for cat, col in zip(categories, sample_colors_from_cmap(num_player_categories))}

    for i in range(len(data.projected_coords)):
        w, h = data.projected_coords[i]
        ax.plot(w, h, 'o', color=player_colors[data.player_categories[i]], markersize=node_size)

    ax.set_axis_off()

    if 'fig' in locals():
        return fig, ax


def plot_tsne_projection(projection, categories=None, plot_colors=None, plot_labels=None, figsize=(9.5, 12), ax=None,
                         num_categories=5):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    if plot_colors is None:
        plot_colors = ['black', 'blue', 'red', 'orange', 'green']

    if plot_labels is None:
        plot_labels = ['Referee', 'Team 1', 'Team 2', 'Goalkeeper A', 'Goalkeeper B']

    if categories is not None:
        for c in range(num_categories):
            ax.scatter(projection[categories == c, 0],
                       projection[categories == c, 1],
                       marker='.',
                       s=12,
                       alpha=1,
                       c=plot_colors[c],
                       label=plot_labels[c])

        lgnd = plt.legend(bbox_to_anchor=(1, 1), fontsize=13)
        for c in range(num_categories):
            lgnd.legendHandles[c]._sizes = [400]
            lgnd.legendHandles[c]._alpha = [1]
    else:
        ax.scatter(projection[:, 0],
                   projection[:, 1],
                   marker='.',
                   s=12,
                   alpha=1)

    plt.tight_layout()
    ax.set_aspect('equal')

    if 'fig' in locals():
        return fig, ax
