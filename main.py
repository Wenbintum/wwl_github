
#%%
from wwl import wwl
from Utility import CollectInputNames, get_catkit_attribute
from Processdata import GraphBase, GraphGroup, CollectDatatoPandas
import os
from Analyzer import KF_validation
import time
import random

database_path   = 'Data/simple_example/'
# feature_list = ['electron_affinity', 'vdw_radius']#,'atomic_number'] #0.280
feature_list = ['atomic_number','band_center'] #'electron_affinity', 'vdw_radius', 'electron_affinity',
# feature_pool= ['atomic_number','en_pauling','atomic_volume','atomic_weight','boiling_point','group_id','period','density'
#  ,'electron_affinity','vdw_radius']
# numbers = [2,3,4,5,6,7,8,9,10]
# for number in numbers:
# collect_time=[]
# collect_number=[]
# for number in [50,100,150,200,250,300,350,400,450,500,550,600]:
# t_time = time.time()
# feature_list =  random.sample(feature_pool, number)
#*single graph test
# file_abspath = os.path.abspath(database_path + 'slab-Cu_111_334#O-mono_2.traj')  #one single example
# import ase.io
# graph_i = GraphBase()
# graph_i.atoms = ase.io.read(file_abspath)
graphs_object   = GraphGroup(database_path, cutoff_mult=1, weighted=False, skin=0.1)
file_names      = CollectInputNames(database_path)
graphs          = graphs_object.Collect_graph()
catkit_pool     = get_catkit_attribute()
node_attributes = graphs_object.Collect_node_attributes(feature_list, attributes_pool=catkit_pool, normalize=False)
ads_energies    = graphs_object.Collect_ads_energies()
datas           = CollectDatatoPandas(graphs, ads_energies, file_names)

# graphs = graphs[:number]; node_attributes=node_attributes[:number]; datas=datas[:number]

#%%
#! to generate different number of configurations
kernel_matrix = wwl(graphs, node_features=node_attributes, num_iterations=1, sinkhorn=False, gamma=None)
print(kernel_matrix)

for gaussian_alpha in [0.001, 0.0006, 0.0008, 0.0005, 0.0004]:
    # kernel_matrix = wwl(graphs, node_features=node_attributes, num_iterations=1, sinkhorn=False, gamma=None)
    KF_validation(kernel_matrix=kernel_matrix, y=datas['target'], name_list=datas['name'], 
                ML_method={'ml':'gpr', 'alpha':gaussian_alpha, 'normalize_y':False},
                    n_split=5, shuffle=True, random_state=0)

    print(KF_validation.avr_train_MAE)
    print(KF_validation.avr_train_RMSE)
    print(KF_validation.avr_test_MAE)
    print(KF_validation.avr_test_RMSE)
    print('\n')
print('time:', round(time.time() - t_time,4), 'number:', number)
collect_number.append(number)
collect_time.append(round(time.time() - t_time,4))
#print('number:', number,  'time:', round(time.time() - t_time,4))
# # from Analyzer import plt_partial
# # plt_partial(KF_validation.test_real, KF_validation.test_pre, KF_validation.test_name)
# from Analyzer import plt_kpca
# plt_kpca(kernel_matrix=kernel_matrix, y=datas['target'], name_list=datas['name'])

#%%
# #* plot distribution
# # from Analyzer import plt_distribution
# # plt_distribution(datas['target'], n_bin=20, dpi=100)
# train_RMSEs,train_MAEs,test_RMSEs,test_MAEs =[],[],[],[]
# for iter in range(1,2):
#     kernel_matrix = wwl(graphs, node_features=node_attributes, num_iterations=iter, sinkhorn=False, gamma=None)
#     KF_validation(kernel_matrix=kernel_matrix, y=datas['target'], name_list=datas['name'], ML_method='gpr',
#                   n_split=5, shuffle=True, random_state=0)
#     train_MAE = KF_validation.avr_train_MAE
#     train_RMSE = KF_validation.avr_train_RMSE
#     test_MAE  = KF_validation.avr_test_MAE
#     test_RMSE = KF_validation.avr_test_RMSE
    
    
#     train_MAEs.append(train_MAE)
#     train_RMSEs.append(train_RMSE)
#     test_MAEs.append(test_MAE)
#     test_RMSEs.append(test_RMSE)
#     print(feature_list)
#     print(train_MAE)
#     print(train_RMSE)
#     print(test_MAE)
#     print(test_RMSE)

#%%
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# x=[1,2,3]
# fig, ax = plt.subplots()
# ax.plot(x, train_MAEs, color='r' ,label='train MAE')
# ax.plot(x, train_RMSEs, color='blue',label='train RMSE')
# ax.plot(x, test_MAEs,'k--', color='r', label='test MAE')
# ax.plot(x, test_RMSEs,'k--', color='blue', label='test RMSE')
# ax.set_ylim(0, 0.5)
# ax.set_ylabel('Error eV')
# # ax.set_xlim(0,0.5)
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1),shadow=True)
# ax.set_xlabel('numer of iteration')
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# # ax.set_xlabel('gamma value')
# # ax.set_xticklabels(['1e-1', '1e-2','1e-4','1e-6','1e-8','1e-10','1e-15','1e-20'])
# # ax.set_ylabel('MAE or RMSE error (eV)')
# plt.savefig('ML_curve.png',dpi=500)
# %%
