'''Visualization functions
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def visualize_embedding_static(Z, class_vec=None, classes=None, special_locs=None, filename=None):
    '''
    Parameters:
      Z: A real-valued two-dimensional array representing the embedding.
        shape = [n_stimuli, n_dim]
      class_vec: An integer array contianing class IDs that indicate
        the class membership of each stimulus.
        shape = [n_stimuli, 1]
      classes: (optional) A dictionary mapping class IDs to strings.
      special_locs: (optional) A boolean array indicating special points to
        emphasize.
        shape = [n_stimuli, 1]
    '''

    # Settings
    dot_size = 20
    cmap = matplotlib.cm.get_cmap('jet')

    [n_stimuli, n_dim] = Z.shape

    n_class = 1
    if class_vec is None:
        use_legend = False

        class_vec = np.ones((n_stimuli))
        unique_class_list = np.unique(class_vec)
        n_class = 1
        color_array = cmap((0,1))
        color_array = color_array[np.newaxis, 0, :]
        
        class_legend = ['all']
    else:
        use_legend = True
        unique_class_list = np.unique(class_vec)
        n_class = len(unique_class_list)
        norm = matplotlib.colors.Normalize(vmin=0., vmax=n_class)
        color_array = cmap(norm(range(12)))

        if classes is not None:
            class_legend = infer_legend(unique_class_list, classes)
        else:
            class_legend = unique_class_list

    if n_dim == 2:
        fig, ax = plt.subplots()
        # Plot each class separately in order to use legend.
        for i_class in range(n_class):
            locs = class_vec == unique_class_list[i_class]
            ax.scatter(Z[locs,0], Z[locs,1], c=color_array[np.newaxis, i_class, :], s=dot_size, label=class_legend[i_class], edgecolors='none')

        if use_legend:
            ax.legend(bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)

        plt.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
        plt.tick_params(
            axis='y',          
            which='both',
            left='off',
            right='off',
            labelleft='off')
        ax.xaxis.get_offset_text().set_visible(False)
    
    if n_dim >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot each class separately in order to use legend.
        for i_class in range(n_class):
            locs = class_vec == unique_class_list[i_class]
            ax.scatter(Z[locs,0], Z[locs,1], Z[locs,2], c=color_array[np.newaxis, i_class, :], s=dot_size, label=class_legend[i_class], edgecolors='none')

        if use_legend:
            ax.legend(bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)

        # plt.axis('off')
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.set_zticks([],[])
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    if filename is None:
        plt.show()
    else:
        # plt.draw()
        plt.savefig(filename, format='pdf', bbox_inches="tight", dpi=100)
    

# def visualize_embedding_movie(Z3, class_vec=None, classes=None, filename=None):
#     '''
#     '''
# TODO

def infer_legend(unique_class_list, classes):
    n_class = len(unique_class_list)
    legend_entries = []
    for i_class in range(n_class):
        class_label = classes[unique_class_list[i_class]]
        legend_entries.append(class_label)    
    return legend_entries
