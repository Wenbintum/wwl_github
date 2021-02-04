
#%%
from wwl import wwl
from Utility import CollectInputNames, get_catkit_attribute
from Processdata import GraphBase, GraphGroup, CollectDatatoPandas
import os
from Analyzer import KF_validation

database_path   = 'Data/Cu_new_201/'
feature_list    = ['atomic_number']#,'atomic_radius']
#*single graph test
# file_abspath = os.path.abspath(database_path + 'slab-Cu_111_334#O-mono_2.traj')  #one single example
# import ase.io
# graph_i = GraphBase()
# graph_i.atoms = ase.io.read(file_abspath)
graphs_object   = GraphGroup(database_path, cutoff_mult=1, weighted=False, skin=0.1)

file_names      = CollectInputNames(database_path)
graphs          = graphs_object.Collect_graph()
catkit_pool     = get_catkit_attribute()
node_attributes = graphs_object.Collect_node_attributes(feature_list, attributes_pool=catkit_pool, normalize=True)
ads_energies    = graphs_object.Collect_ads_energies()
datas           = CollectDatatoPandas(graphs, ads_energies, file_names)


kernel_matrix = wwl(graphs, node_features=node_attributes, num_iterations=1, sinkhorn=False, gamma=None)
KF_validation(kernel_matrix=kernel_matrix, y=datas['target'], name_list=datas['name'], ML_method='gpr',
                n_split=5, shuffle=True, random_state=0)
from Analyzer import plt_partial
plt_partial(KF_validation.test_real, KF_validation.test_pre, KF_validation.test_name)

#%%
#* plot distribution
# from Analyzer import plt_distribution
# plt_distribution(datas['target'], n_bin=20, dpi=100)
train_RMSEs,train_MAEs,test_RMSEs,test_MAEs =[],[],[],[]
for iter in range(1,2):
    kernel_matrix = wwl(graphs, node_features=node_attributes, num_iterations=iter, sinkhorn=False, gamma=None)
    train_MAE, train_RMSE, test_MAE, test_RMSE, test_outliers = \
    KF_validation(kernel_matrix=kernel_matrix, y=datas['target'], name_list=datas['name'], ML_method='gpr',
                  n_split=5, shuffle=True, random_state=0)
    train_MAEs.append(train_MAE)
    train_RMSEs.append(train_RMSE)
    test_MAEs.append(test_MAE)
    test_RMSEs.append(test_RMSE)
    print(feature_list)
    print(train_MAE)
    print(train_RMSE)
    print(test_MAE)
    print(test_RMSE)

#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x=[1,2,3]
fig, ax = plt.subplots()
ax.plot(x, train_MAEs, color='r' ,label='train MAE')
ax.plot(x, train_RMSEs, color='blue',label='train RMSE')
ax.plot(x, test_MAEs,'k--', color='r', label='test MAE')
ax.plot(x, test_RMSEs,'k--', color='blue', label='test RMSE')
ax.set_ylim(0, 0.5)
ax.set_ylabel('Error eV')
# ax.set_xlim(0,0.5)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1),shadow=True)
ax.set_xlabel('numer of iteration')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# ax.set_xlabel('gamma value')
# ax.set_xticklabels(['1e-1', '1e-2','1e-4','1e-6','1e-8','1e-10','1e-15','1e-20'])
# ax.set_ylabel('MAE or RMSE error (eV)')
plt.savefig('ML_curve.png',dpi=500)
# %%
