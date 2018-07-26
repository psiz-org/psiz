"""Tempory file for quick testing."""
import copy
import numpy as np

from psiz.utils import similarity_matrix, possible_outcomes, matrix_correlation
from psiz.trials import UnjudgedTrials
from psiz.models import Exponential


def main():
    """Main."""
    model = ground_truth()

    def similarity_func1(z_q, z_ref):
        return model.similarity(z_q, z_ref, attention=model.phi['phi_1']['value'][0])
    simmat1 = similarity_matrix(similarity_func1, model.z['value'])

    def similarity_func2(z_q, z_ref):
        return model.similarity(z_q, z_ref, attention=model.phi['phi_1']['value'][1])
    simmat2 = similarity_matrix(similarity_func2, model.z['value'])

    print('here')


def ground_truth():
    """Return a ground truth model."""
    n_dim = 2
    n_group = 2

    model = Exponential(3, n_dim, n_group)
    z = np.array((
        (.1, .1), (.15, .2), (.4, .5)
    ))
    attention = np.array((
        (1.2, .8),
        (.7, 1.3)
    ))
    freeze_options = {
        'z': z,
        'theta': {
            'rho': 2,
            'tau': 1,
            'beta': 10,
            'gamma': 0
        },
        'phi': {
            'phi_1': attention
        }
    }
    model.freeze(freeze_options)
    return model


if __name__ == "__main__":
    main()