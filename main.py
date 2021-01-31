
#%%
from wwl import wwl
from Utility import CollectInputNames
database_path = 'Data/Cu_718_filter/'
file_names  = CollectInputNames(database_path)



#%%
file_names  = CollectInputData(database_path)

graphs = Collect_graph(database_path, cutoff_mult=1, weighted=False, skin=0.1)
catkit_attr = get_catkit_attribute()  #catkit feature library 


feature_list  = ['Local_parameter', 'vdw_radius', 'electron_affinity']
# feature_list = ['atomic_number']
# feature_list = ['PE', 'EA', 'IE', 'GP']
#feature_list = ['atomic_number','atomic_volume','electron_affinity',
                #'en_pauling' , 'dband_center_bulk', 'heat_of_formation', 'lattice_constant', 'dband_width_bulk']

node_attributes = Collect_node_attributes(database_path, feature_list=feature_list, 
                                        attr_pool=catkit_attr, normalize=True)
ads_energies = Collect_ads_energies(database_path)
datas = CollectDatatoPandas(graphs,ads_energies,file_names)

    # plt_distribution(datas['target'], n_bin=20, dpi=100)
    # train_RMSEs,train_MAEs,test_RMSEs,test_MAEs =[],[],[],[]
    
train_RMSEs,train_MAEs,test_RMSEs,test_MAEs =[],[],[],[]
for iter in range(1,2):
    kernel_matrix = wwl(graphs, node_features=node_attributes, num_iterations=iter, sinkhorn=False, gamma=None)
    train_MAE, train_RMSE, test_MAE, test_RMSE, test_outliers = \
    KF_validation(kernel_matrix=kernel_matrix, y=datas['target'], name_list=datas['name'], ML_method='gpr',
                n_split=5,shuffle=True,random_state=0)
    train_MAEs.append(train_MAE)
    train_RMSEs.append(train_RMSE)
    test_MAEs.append(test_MAE)
    test_RMSEs.append(test_RMSE)
    print(feature_list)
    print(train_MAE)
    print(train_RMSE)
    print(test_MAE)
    print(test_RMSE)

# # %%
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# x=[i for i in range(50)]
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
