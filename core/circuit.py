import pennylane as qml
import torch
import numpy as np
from typing import List
from .layers import *
from .utils.arguments import circuitarguments

pi = np.pi

# Quantum circuit class definition
class Circuit:

    def __init__(self, params: circuitarguments):
        
        # Circuit configuration
        self.p = params

        # Layer container
        self.layers : List[Entangler] = []

        # Deprecated
        self.init = None

        # Post selection layer
        self.ps = None

        # Pennylane device class for qnode
        self.dev = qml.device('default.mixed', wires = self.p.num_wires)
        self.circuit = None
        self.bound = []

        # Circuit build
        pivot = 0
        if self.p.num_wires == 1:
            self.init = Initialization(self.p.num_wires, 'Init', 'X')
            self.ramsey = RamseyZ(self.p.num_wires, 'Ramsey', pivot, self.p.t2, self.p.p, self.p.gm_ratio)
            pivot += self.ramsey.n_params
            b = self.ramsey.get_param_bound()
            if b is not None:
                self.bound.extend(b)
            
        else:
            self.init = Initialization(self.p.num_wires, 'Init', 'Y')

            for i in range(self.p.num_entangler):
                l = Entangler(self.p.num_wires, f'Entangler_{i+1}', pivot, self.p.t2)
                pivot += l.n_params
                self.layers.append(l)
                b = l.get_param_bound()
                if b is not None:
                    self.bound.extend(b)

            self.ramsey = RamseyZ(self.p.num_wires, 'Ramsey', pivot, self.p.t2, self.p.p, self.p.gm_ratio)
            b = self.ramsey.get_param_bound()
            if b is not None:
                self.bound.extend(b)
            pivot += self.ramsey.n_params

        @qml.qnode(self.dev)
        def circuit(B: np.array, w: np.array):
            
            self.init()

            for l in self.layers:
                l(w)

            if isinstance(self.ramsey, Ramsey):
                self.ramsey(B)
            else:
                self.ramsey(w, B)
            
            return qml.density_matrix(wires=range(self.p.num_wires))

        if self.p.ps:
            self.ps = PostSelection(self.p.num_wires, 'Post_Selection', pivot)
            pivot += self.ps.n_params

            b = self.ps.get_param_bound()
            if b is not None:
                self.bound.extend(b)

        self.n_params = pivot
        bound_np = np.array(self.bound)
        
        rng = np.random.default_rng(seed=42)
        self.w = rng.uniform(bound_np[:,0], bound_np[:,1])
        
        if self.ps:
            @qml.qnode(self.dev)
            def circuit_ps(B: np.array, w:np.array):

                self.ps(circuit(B, w), w)
                return qml.density_matrix(wires=range(self.p.num_wires))
            
            self.circuit = circuit_ps
            self.inner_circuit = circuit

        else:
            self.circuit = circuit

    def view_param(self):
        
        for l in self.layers:
            print(f'{l.id} = {self.w[l.offset:l.offset+l.n_params]}')
        if isinstance(self.ramsey, RamseyZ):
            print(f'{self.ramsey.id} = {self.w[self.ramsey.offset:self.ramsey.offset+self.ramsey.n_params]}')
        
        if self.ps:
            print(f'{self.ps.id} = {self.w[self.ps.offset:self.ps.offset+self.ps.n_params]}')


    def fmod_param(self):

        self.w = np.fmod(self.w,2*pi)

    
    def draw_circuit(self):

        if self.ps:
            fig, ax = qml.draw_mpl(self.inner_circuit)(np.array([0.0]))
            fig.savefig('./circuit.png')
        else:
            fig, ax = qml.draw_mpl(self.circuit)(np.array([0.0]))
            fig.savefig('./circuit.png')

    
    def plot_params(self, data):

        for l in self.layers:
            l.plot_data(data)
        self.ramsey.plot_data(data)
        if self.ps is not None:
            self.ps.plot_data(data)

    
    def set_params(self, x):

        self.w = x