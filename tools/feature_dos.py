#file_i: the name of trajectory file
#the features are calculated separately by slab and adsorbate step by step
slab_name = file_i.split('#')[0].split('-')[1]
ads_name  = file_i.split('#')[1].split('-')[0]
dos_pathway = '/home/wenbin/Calculation/graph_project/dos/'
#atom index is slab is same as clean surface
#for i, atom_index in enumerate(atoms_indexs) i is the index of atom_i
#extract clean surface feature.
import os 
import glob
import pickle

#clean surface
with open(dos_pathway+slab_name+'.pickle', 'rb') as input_file:
    dos_energies, dos_total, pdos = pickle.load(input_file)
    #calculate pdos for each atom of atoms
    for atom_index in atoms_indexs:
        
        if atoms[atom_index].number == 6 or atoms[atom_index].number == 8:
            states = 'p'
            summed_pdos = pdos[atom_index][states][0] + pdos[atom_index][states][1]
        elif atoms[atom_index].number == 1:
            states = 's'
            summed_pdos = pdos[atom_index][states][0] + pdos[atom_index][states][1]
        elif atoms[atom_index].number > 18:
            #Ni is spin calculation
            states = 'd'
            if atoms[atom_index].symbol in ['Fe', 'Co', 'Ni']:
                summed_pdos = pdos[atom_index][states][0] + pdos[atom_index][states][1]
            else:
                summed_pdos = pdos[atom_index][states][0]
        else:
            raise TypeError('unidentified atom species')
        
        band_center = 
        
        

    


def band_center():
    
    
