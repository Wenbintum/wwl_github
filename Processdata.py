from ase import neighborlist
#from ase.data import atomic_numbers, atomic_masses, covalent_radii
from ase.io import read, write
import numpy as np
from Utility import Constant,Adsorption_energy,CollectInputData,get_node_attribute, Spherical_hamonics_1To6
import igraph as ig
import pandas as pd
from ase.data import covalent_radii
import os

def Node_representation(atoms, cutoff_mult=1, skin=0.3):    #adjacency matrix
    cutoff       = neighborlist.natural_cutoffs(atoms,mult=cutoff_mult)
    neighborList = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True, skin=skin)
    neighborList.update(atoms)
    node_matrix  = neighborList.get_connectivity_matrix(sparse=False).astype(float)
    return node_matrix

def Create_graph(adj_mat):
    g = ig.Graph.Adjacency(adj_mat.tolist(), mode='undirected')
    return g

def Collect_graph(database_path, cutoff_mult=1, weighted = False, skin=0.3):
    graphs = []
    file_list  = CollectInputData(database_path)
    for i, file_i in enumerate(file_list):
        atoms = read(database_path + file_i)
        #DeleteFixAtoms(atoms) #wenbin
        adj_matrix = Node_representation(atoms, cutoff_mult=cutoff_mult, skin=skin)
        graph = Create_graph(adj_mat=adj_matrix)
        if weighted == True:
            graph = Assign_Edge_weight(atoms,graph,database_path+file_i)
        graphs.append(graph)
    return graphs
def Cal_Edge_weight(atoms, atom_i, atom_j):
    if atom_i == atom_j:
        return 1
    else:
        bond_length = atoms.get_distance(atom_i,atom_j,mic=True)
        covalent_r = covalent_radii[atoms[atom_i].number] + covalent_radii[atoms[atom_j].number]
        return covalent_r/bond_length

def Define_ads_bond(atoms, graph):
    ads_bond_index = []
    for node_i in graph.vs.indices:
        if atoms[node_i].number < 18:  
           for edge_i in graph.es[graph.incident(node_i)]:
               if atoms[edge_i.source].number > 18 or  atoms[edge_i.target].number > 18 :
                   ads_bond_index.append(node_i)
    return list(set(ads_bond_index))

def Assign_Edge_weight(atoms, graph, file_name):
    tags=False
    edge_list = graph.get_edgelist()
    ads_bond_index = Define_ads_bond(atoms,graph)   #define bonding atoms
    for e_index, edge_i in enumerate(edge_list):
        #print(e_index, edge_i)
        weight_i_list = []
        for ads_atom_i in ads_bond_index:    #cal contribution of source and target of edge in reference with ads_atoms
            weight_i_list.append(Cal_Edge_weight(atoms,edge_i[0], ads_atom_i))
            weight_i_list.append(Cal_Edge_weight(atoms,edge_i[1], ads_atom_i))
        try:
            weight_i = sum(weight_i_list)/len(weight_i_list)
        except:
            tags = True
            weight_i = 0
        graph.es[e_index]['weight'] = weight_i
    if tags == True:
        print(atoms)
        print(file_name)
        os.system('mv {} {}'.format('/home/wenbin/Software/VScode/wwl_test/'+ file_name,'/home/wenbin/Software/VScode/wwl_test/tmp/'))
    return graph

def Node_attributes(atoms, feature_list,attr_pool,cutoff_mult=1,skin=0.1, file_i=None):
    atom_indexs = atoms.get_atomic_numbers()
    node_attributes = {}
    #
    adj_matrix = Node_representation(atoms, cutoff_mult=cutoff_mult, skin=skin)
    graph = Create_graph(adj_mat=adj_matrix)
    
    for i, atom_index in enumerate(atom_indexs):
        node_attributes[i] = []
        for feature in feature_list: #feature by feature for one structure
            if feature in attr_pool.keys():
                node_attributes[i].append(get_node_attribute(atom_index,feature,attr_pool))
            elif feature == 'Local_parameter':
                node_attributes[i].extend(Spherical_hamonics_1To6(i, atoms, graph, file_i))
            else:
                raise NameError('UNKNOWN FEATURE')
            # print(feature)
        #print(node_attributes[i])
    return np.array(list(node_attributes.values()))
    
    
    # chemical_symbols = atoms.get_chemical_symbols()
    # node_attributes = {}
    # for i, chemical_symbol in enumerate(chemical_symbols):
    #     node_attributes[i] = []
    #     for feature in feature_list:
    #         node_attributes[i].append(Constant[chemical_symbol][feature])
    # return np.array(list(node_attributes.values()))        

def Collect_node_attributes(database_path,feature_list,attr_pool,normalize=True):
    sum_node_attributes = []
    file_list = CollectInputData(database_path)
    for i, file_i in enumerate(file_list):
        atoms = read(database_path + file_i)
        #atoms = DeleteFixAtoms(atoms) #wenbin
        attribute_graph_i = Node_attributes(atoms=atoms,feature_list=feature_list,attr_pool=attr_pool,file_i=file_i)
        sum_node_attributes.append(attribute_graph_i)
    if normalize:
        return NormalizeAttribute(sum_node_attributes)
    else:
        return np.array(sum_node_attributes)    

def Collect_ads_energies(database_path):
    ads_energies = []
    file_list = CollectInputData(database_path)
    for i, file_i in enumerate(file_list):
        ads_energy = Adsorption_energy(file_i)
        ads_energies.append(ads_energy)
    return ads_energies
    
def CollectDatatoPandas(graphs,target,name=None):
    column_names = ['graphs','target','name']
    datas = pd.DataFrame(columns=column_names)
    datas['graphs'] = graphs
    datas['target'] = target
    datas['name']   = name
    return datas

def NormalizeAttribute(node_attributes):
    conc_attributes = np.concatenate(tuple(i for i in node_attributes),axis=0)
    normed_conc_attributes = conc_attributes / np.nanmax(np.abs(conc_attributes),axis=0) # [-1:1] #conc_attributes / conc_attributes.max(axis=0) [0:1]
    indexlist = np.cumsum(list(map(lambda node_attribute: len(node_attribute), node_attributes)))
    #indexlist = np.cumsum(lengthlist)
    # normed_conc_attributes[np.isnan(normed_conc_attributes)] = 2 #sign an outlier value for nan 
    normed_node_attributes = np.array_split(normed_conc_attributes, indexlist[:-1])

    return np.array(normed_node_attributes)

def DeleteFixAtoms(atoms):
    fix_indexs = atoms.constraints[0].get_indices()
    del atoms[fix_indexs]
    return atoms
    