from abc import ABCMeta, abstractmethod

def main():
    model = Exponential()
    
    params = {'rho': 2.5, 'tau': 1.5, 'beta': 1.5, 'gamma': 9}

    model.set_params(params)
    print('here')

class PsychologicalEmbedding(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
    
    def set_params(self, params):
        for param_name in params:
            setattr(self, param_name, params[param_name])

    # def set_params(self, params):
    #     pass

class Exponential(PsychologicalEmbedding):

    def __init__(self):
        """
        """
        self.rho = 2.
        self.tau = 1.
        self.gamma = 0.
        self.beta = 10.

    # def set_params(self, params):
    #     self.rho = params['rho']
    #     self.tau = params['tau']
    #     self.beta = params['beta']
    #     self.gamma = params['gamma']    

if __name__ == "__main__":
    main()