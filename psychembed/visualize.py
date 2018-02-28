'''Visualization functions
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def visualize_embedding_static(Z, class_vec=None, classes=None, special_locs=None):
    '''
    Parameters:
      Z: A real-valued two-dimensional array representing the embedding.
        shape = [n_stimuli, n_dim]
      class_vec: (optional) A integer array contianing class IDs that indicate
        the class membership of each stimulus.
        shape = [n_stimuli, 1]
      classes: (optional) A dictionary mapping class IDs to strings.
      special_locs: (optional) A boolean array indicating special points to
        emphasize.
        shape = [n_stimuli, 1]
    '''

    # Settings
    dot_size = 20

    [n_stimuli, n_dim] = Z.shape

    if class_vec is None:
        colors = np.ones((n_stimuli))
    else:
        colors = class_vec

    if n_dim == 2:
        fig = plt.figure()
        ax = plt.scatter(Z[:,0], Z[:,1], s=dot_size, c=colors)

        plt.show()


    if n_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=dot_size, c=colors)

        # plt.axis('off')
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.set_zticks([],[])
        plt.show()

# def visualize_embedding_movie(Z3, class_vec, classes, filename):
#     '''
#     '''
