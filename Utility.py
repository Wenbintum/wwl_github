from ase.io import read, write
import numpy as np
import os
from copy import deepcopy
import pkg_resources
import json
from scipy.special import sph_harm
import warnings

#reference energy: adsorbate + clean surface
#adsorbate will be calculated by the numbers of atomic numer in Atoms project   
Eref={'CH3OH': -693.7189669146002,
      'CO'   : -626.4867272396203,
      'H2O'  : -496.2714430827127,
      'H'    : -16.80805991874496,   #0.25*(E(CH3OH) - E(CO))
      'O'    : -462.6553232452228,   #E(H2O) - 2*E(H)
      'C'    : -163.83140399439753   #E(CO) - E(O)
      }

def CollectInputNames(folder):
    file_list = os.listdir(folder)
    copy_file_list = deepcopy(file_list)
    for file_i in copy_file_list:
        if '#' not in file_i:
            file_list.remove(file_i)
    return file_list
            
def Ref_adsorbate(database_path,structure):
    """
    docstring
    """
    global Eref
    atoms = read(database_path + structure)
    numbers_list = atoms.get_atomic_numbers()
    for i in numbers_list:
        if i < 19 and i !=1 and i !=6 and i!=8:
            raise TypeError('not only including C, H and O')
    #number of H, C, O
    nH = np.count_nonzero(numbers_list==1)
    nC = np.count_nonzero(numbers_list==6)
    nO = np.count_nonzero(numbers_list==8)
    Ereference = nH * Eref['H'] + nC * Eref['C'] + nO * Eref['O']
    return Ereference

def Ref_cleansurface(database_path,structure):
    """[summary]

    Args:
        structure ([type]): [description]

    Raises:
        NameError: [description]

    Returns:
        [type]: [description]
    """
    slab_name = structure.split('#')[0].split('-')[1]
    traj_name = database_path + slab_name + '.traj'
    if not os.path.exists(traj_name):
        raise NameError('Can not find {}'.format(traj_name))
    atoms=read(traj_name)
    Ereference = atoms.get_potential_energy()
    return Ereference
    
def Total_surface(database_path, structure):
    atoms=read(database_path+structure)
    return atoms.get_potential_energy()

def Adsorption_energy(database_path, structure):
    return Total_surface(database_path, structure) - Ref_adsorbate(database_path, structure) -\
           Ref_cleansurface(database_path, structure)

def get_catkit_attribute():
    path = pkg_resources.resource_filename('catkit', 'data/properties.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data




#list of node attribute
#atomic numbers: AN, atomic radius: AR, d-band properties of pure metals, dbp
#electron affinity: EA,  ionization potential, IP,  Dipole polarizability, DP
# heat of formation, HE,  electronegativitym  PE,  coordination number, CN, van der waals, VDW

def Spherical_hamonics_1To6(atom_index, atoms, graph, file_i):   #l= 1,2,3,4,5,6
    neighbors_coordinate=[]
    print(file_i, atom_index)
    for neighbor_i in graph.neighbors(atom_index):
        neighbors_coordinate.extend(atoms[neighbor_i].position)
    n_neighbors=len(graph.neighbors(atom_index))
    if n_neighbors == 0:    #treat atom with a bit large distance as a single bond atom
        raise ValueError('atom with 0 neighbor exists')
    else:
        #determine the coordinate of nearby Oxygen atom, by edge to find neighbor atoms
        neighbors_coordinate = np.reshape(neighbors_coordinate, (n_neighbors,3))       
        center_coordinate = atoms[atom_index].position
        #get vector by Oxygen list minus active site coordinate i.g. vector ri
        vec2_multi = neighbors_coordinate - center_coordinate
        #compute phi: polar angle  and   theta: azimuth angle
        phi_list=[]
        theta_list=[]
        for i in range(n_neighbors):
            r=np.sqrt(vec2_multi[i][0]**2 + vec2_multi[i][1]**2 + vec2_multi[i][2]**2)
            phi=np.arccos(vec2_multi[i][2]/r)
            theta=np.arctan2(vec2_multi[i][1],vec2_multi[i][0])
            if theta < 0:
                theta=theta+2*np.pi
            phi_list.append(phi)
            theta_list.append(theta)
        #phi_list=[round(i*180/np.pi,2) for i in phi_list]
        #compute spherical hamonics
        Q_l_list=[]
        for l in range(1,7):    #l=1-6  
            Q_lms=0
            for m in range(-l,l+1):
                Y_lms=0
                for i in range(n_neighbors):
                    Y_lm= sph_harm(m,l,theta_list[i],phi_list[i])
                    Y_lms += Y_lm
                Q_lm=Y_lms/n_neighbors #length of oxygen_list
                Q_lms=Q_lms+ abs(Q_lm)**2
            Q_l=np.sqrt(4*np.pi/(2*l+1)*Q_lms)
            Q_l_list.append(round(Q_l,4))
        return Q_l_list[0], Q_l_list[1], Q_l_list[2], Q_l_list[3], Q_l_list[4], Q_l_list[5]

if __name__ == "__main__":
    print(Constant)
