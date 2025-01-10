import numpy as np

class MLP:
    """
    Class to define Multilayer Perceptrons.
    Declare instance with MLP(layers).
    """
    def __init__(self, layers: tuple):
        """
        layers: a tuple with (ninputs, nhidden1, nhidden2, ... noutput)
        """
        self.layers = layers
        self.trace = False
        self.threshold = 5.0
        self.labels = None # text for the labels [input-list, output-list]
        
        self.size = 0
        self.W = [] # list of numpy matrices
        self.b = [] # list of numpy vectors
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i],layers[i+1])-0.5
            b = np.random.rand(layers[i+1])-0.5
            self.W.append(w)
            self.b.append(b)
            self.size += layers[i] * layers[i+1] + layers[i+1]
        
        self.lRMS = [] # hold all traced RMSs to draw graph
        self.laccuracy = [] # hold all traced accuracies to draw graph
        
    def sigm(self, neta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-neta))
    
    def forward(self, x): # fast forward (optimized in time, but not use to train!)
        for i in range(len(self.b)):
            net = np.dot(x,self.W[i]) + self.b[i]
            x = self.sigm(net)
        return x
            
    def to_chromosome(self):
        '''
        Convert weights and biases to a flatten list to use in AG.
        '''
        ch = []
        for w,b in zip(self.W,self.b):
            ch += w.flatten().tolist()
            ch += b.flatten().tolist()
        return ch
    
    def from_chromosome(self, chromosome: list):
        '''
        Convert a flatten list (chromosome from a GA) to internal weights and biases.
        '''
        if len(chromosome) != self.size:
            print(self.size)
            raise ValueError("Chromosome legnth doesn't match architecture")
        self.W = []
        self.b = []
        pos = 0
        for i in range(len(self.layers)-1): # for each layer
            to = self.layers[i]*self.layers[i+1] # number of weight
            w = np.array(chromosome[pos:pos+to]).reshape(self.layers[i],self.layers[i+1])
            pos += to
            to = self.layers[i+1] # number of bias
            b = np.array(chromosome[pos:pos+to]).reshape(self.layers[i+1])
            pos += to
            
            self.W.append(w)
            self.b.append(b)
