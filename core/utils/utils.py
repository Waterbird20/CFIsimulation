# utility methods
import os
import pennylane as qml
from itertools import combinations
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# Entangler Hamiltonian getter
def get_entangler(num_wires):

    if num_wires < 2:
        raise ValueError('Number of wires must be greater than 1')
    
    H = []
    for entry in combinations(range(num_wires), 2):
        H.append(qml.PauliZ(entry[0]) @ qml.PauliZ(entry[1]))

    return qml.Hamiltonian(coeffs = [0.5]*len(H), observables = H)


# Ramsey free-evolution hamiltonian getter
def get_ramsey(num_wires, gm_ratio, t_obs):

    H = []
    for i in range(num_wires):
        H.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs = [gm_ratio*t_obs*0.5]*len(H), observables = H)


def plot_density_matrix(rho, t, save_as):

    fig = plt.figure(figsize=(10, 8))


    ax1 = fig.add_subplot(111, projection='3d')
    n = rho.shape[0]
    xpos, ypos = np.meshgrid(np.arange(n), np.arange(n))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.8
    dz_real = rho.real.flatten()
    colors_real = plt.cm.rainbow((dz_real - dz_real.min()) / (dz_real.max() - dz_real.min()))
    
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz_real, color=colors_real, shade=True, alpha=0.7)
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_zlabel('Amplitude')
    ax1.set_title(f'Real Part of Density Matrix at t = {t:.6f}')
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.view_init(elev=50, azim=30)
    
    norm = Normalize(vmin = dz_real.min(), vmax = dz_real.max())
    sm = ScalarMappable(norm=norm, cmap='rainbow')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, pad=0.1, shrink=0.5, aspect=10)  # 위치와 크기 조정
    cbar.set_label('Amplitude')

 
    # ax2 = fig.add_subplot(122, projection='3d')
    # dz_imag = rho.imag.flatten()
    # colors_imag = plt.cm.rainbow((dz_imag - dz_imag.min()) / (dz_imag.max() - dz_imag.min()))
    
    # ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_imag, color=colors_imag, shade=True, alpha=0.7)
    # ax2.set_xlabel('Column')
    # ax2.set_ylabel('Row')
    # ax2.set_zlabel('Amplitude')
    # ax2.set_title('Imaginary Part of Density Matrix')
    # ax2.set_xticks(range(n))
    # ax2.set_yticks(range(n))
    # ax2.view_init(elev=50, azim=30)

    plt.tight_layout()
    plt.savefig(f'./dmplots/{save_as}.png')
    plt.close()


# Dephasing coefficient calculator
def dephase_factor(tau):
    return 1 - torch.exp(-2*tau)

def dephase_factor_nontorch(tau):
    return 1 - np.exp(-2*tau)

# (Test) Depolarization channel
def get_noise_channel(tau):

    channel_I = np.sqrt(1 - 3*tau/4)*np.array([[1.,0.],[0.,1.]])
    channel_X = np.sqrt(tau/4)*np.array([[0.,1.],[1.,0.]])
    channel_Y = np.sqrt(tau/4)*np.array([[0.,-1.j],[1.j,0.]])
    channel_Z = np.sqrt(tau/4)*np.array([[1.,0.],[0.,-1.]])

    return [channel_I, channel_X, channel_Y, channel_Z]


def clean_container():

    for entry in os.listdir('./dmplots'):
        os.remove(f'./dmplots/{entry}')
    for entry in os.listdir('./paramplots'):
        os.remove(f'./paramplots/{entry}')
    try:
        os.remove('./data.npy')
    except:
        pass

def parse_data(save_to):
    os.system(f'mkdir {save_to}')
    os.system(f'cp ./{save_to}_data.npy ./{save_to}/{save_to}_data.npy')
    files = [f for f in os.listdir('./paramplots')]
    for filename in files:
        os.system(f'cp ./paramplots/{filename} ./{save_to}/{filename}')
    #os.system(f'tar -czf ./{save_to}.tar.gz ./paramplots ./data.npy ./circuit.png ./config.yaml')