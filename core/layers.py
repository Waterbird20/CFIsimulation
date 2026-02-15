# Layer class definitions
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils.utils import get_entangler, get_ramsey, dephase_factor, dephase_factor_nontorch
from functools import reduce

pi = np.pi

# Base class for layer
class CircuitLayer:

    def __init__(self, num_wires, layer_id):

        self.num_wires = num_wires
        self.id = layer_id
        self.n_params = 0
        self.data_label = None
        self.offset = None
        self.bound = None


    def plot_data(self, data):

        pass


    def get_param_bound(self):
        return self.bound


class Initialization(CircuitLayer):

    def __init__(self, num_wires, layer_id, init_type):
        
        if init_type not in 'XY':
            raise ValueError('Invalid Initialization type')
        
        super().__init__(num_wires, layer_id)
        self.init_type = init_type


    def __call__(self):
        
        if self.init_type == 'X':
            for i in range(self.num_wires):
                qml.RX(pi/2, wires=i)

        elif self.init_type == 'Y':
            for i in range(self.num_wires):
                qml.RY(pi/2, wires=i)

# Entangler layer
class Entangler(CircuitLayer):

    def __init__(self, num_wires, layer_id, offset, t2):

        super().__init__(num_wires, layer_id)

        self.offset = offset
        self.t2 = t2
        self.n_params = 3
        self.gamma = 1.4e+7
        self.H = get_entangler(num_wires)
        self.data_label = [r'$\tau$', r'$\theta$', r'${\tau}^{\prime}$']
        self.bound = [(-2*pi, 2*pi)]*self.n_params


    def __call__(self, w):

        o = self.offset
    
        qml.ApproxTimeEvolution(self.H, w[o], 1)
        for i in range(self.num_wires):
            qml.RX(w[o+1], wires=i)
            qml.RY(-pi/2, wires=i)

        qml.ApproxTimeEvolution(self.H, w[o+2], 1)
        for i in range(self.num_wires):
            qml.RY(pi/2, wires=i)

# Ramsey free evolution layer with RZ gate 
class Ramsey(CircuitLayer):

    def __init__(self, num_wires, layer_id, t2, gm_ratio, B):

        super().__init__(num_wires, layer_id)

        self.H = get_ramsey(num_wires, gm_ratio, B)
        self.t2 = t2

    
    def __call__(self, t):

        tau = 1 - np.exp(-2*t.item()/self.t2)
        channel_I = np.sqrt(1 - 3*tau/4)*np.array([[1.,0.],[0.,1.]])
        channel_X = np.sqrt(tau/4)*np.array([[0.,1.],[1.,0.]])
        channel_Y = np.sqrt(tau/4)*np.array([[0.,-1.j],[1.j,0.]])
        channel_Z = np.sqrt(tau/4)*np.array([[1.,0.],[0.,-1.]])

        qml.ApproxTimeEvolution(self.H, t, 1)
        
        for i in range(self.num_wires):
            # qml.QubitChannel([channel_I, channel_Z, channel_X, channel_Y], wires=i)
            qml.RX(pi/2, wires=i)

    def get_param_bound(self):
        return None

# Ramsey free evolution layer with RZ gate at the end
class RamseyZ(CircuitLayer):

    def __init__(self, num_wires, layer_id, offset, t2, p, gm_ratio):

        super().__init__(num_wires, layer_id)
        self.t2 = t2
        self.gm_ratio = gm_ratio
        self.offset = offset
        self.n_params = self.num_wires + 1
        self.p = p
        self.data_label = [r'$t_{s}$'] + [rf'$\theta_{{Z{i+1}}}$' for i in range(self.num_wires)]
        self.bound = [(1e-9,2*self.t2)] + [(-2*pi, 2*pi)]*(self.num_wires)

    
    def __call__(self, w, B):

        o = self.offset
        phi = np.abs(w[o])/self.t2
        if phi < 0:
            print(phi)
            print(w[o], o)
        H = get_ramsey(self.num_wires, self.gm_ratio, w[o])
        tau = dephase_factor_nontorch((phi)**(self.p))
        qml.ApproxTimeEvolution(H, B, 1)
        
        for i in range(self.num_wires):
            qml.PhaseDamping(tau, wires=i)
            qml.RZ(w[o+i+1], wires=i)
            qml.RX(pi/2, wires=i)

    def get_param_bound(self):
        return super().get_param_bound()

# Post selection layer
class PostSelection(CircuitLayer):

    def __init__(self, num_wires, layer_id, offset):
        
        super().__init__(num_wires, layer_id)
        self.n_params = 3*num_wires
        self.offset = offset
        self.data_label = [rf'$\gamma_{{{i+1}}}$' for i in range(self.num_wires)]
        for i in range(self.num_wires):
            self.data_label.append(rf'$\theta_{{Z{i+1}}}$')
            self.data_label.append(rf'$\theta_{{X{i+1}}}$')
        self.bound = [(0,1-1e-3)]*self.num_wires + [(-2*pi, 2*pi)]*(2*self.num_wires)


    def __call__(self, rho, w):
        
        o = self.offset
        ps = []
        
        for i in range(self.num_wires):
            ps.append(np.array([[np.sqrt(1-np.clip(w[o+i],0,0.999)),0.],[0,1]],dtype=np.complex128))
        K = reduce(lambda x,y: np.kron(x,y), ps)
        numerator = K @ rho @ K.conj().T
        denominator = np.trace(numerator)

        rho_ps = numerator / denominator

        qml.QubitDensityMatrix(rho_ps, wires=range(self.num_wires))
        for i in range(self.num_wires):
            qml.RX(w[o+i*2 + self.num_wires], wires=i)
            qml.RZ(w[o+i*2 + self.num_wires+1], wires=i)

# Deprecated
class Rotate(CircuitLayer):

    def __init__(self, num_wires, layer_id, offset, axis):

        if axis not in 'XYZ':
            raise TypeError('Invalid axis type')
        
        super().__init__(num_wires, layer_id)
        self.axis = axis

        self.offset = offset
        self.n_params = 1
        self.data_label = [r'$\theta$']

    
    def __call__(self, w):
        
        o = self.offset

        if self.axis == 'X':
            for i in range(self.num_wires):
                qml.RX(w[o], wires=i)
        elif self.axis == 'Y':
            for i in range(self.num_wires):
                qml.RY(w[o], wires=i)
        else:
            for i in range(self.num_wires):
                qml.RZ(w[o], wires=i)