import numpy as np

class PFVectorMemory(object):

    def __init__(self, ticker_num, beta_pvm, training_steps, training_batch_size, wt_vector_init):
        
        #pvm at all times
        self.pvm = np.transpose(np.array([wt_vector_init] * int(training_steps)))
        self.beta_pvm = beta_pvm
        self.training_steps = training_steps
        self.training_batch_size = training_batch_size

    def get_wt_vector_t(self, t):
        return self.pvm[:, t]

    def update_wt_vector_t(self, t, weight):
        self.pvm[:, int(t)] = weight
            
    def test(self):
        return self.pvm