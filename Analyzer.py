
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from ML_learning import Train_gpr, Train_krr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statistics import mean
import plotly.graph_objects as go
from sklearn.decomposition import KernelPCA

def plt_distribution(x, n_bin, dpi=100):
    fig, axs = plt.subplots(tight_layout=True)
    #fig, axs = plt.subplots(1, 2, tight_layout=True) #sharey=True
    axs.hist(x, bins=n_bin)
    axs.set_xlabel('Adsorption energy (eV)')
    axs.set_ylabel('Numbers')
    # axs[1].hist(x, bins=n_bin, density=True)
    # axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.savefig('Result/distribution.png',dpi=dpi)

def Check_outlier(target,prediction,name_list,threshold=0.5):
    outlier_list = {}
    for i, values in enumerate(list(zip(target,prediction,name_list))):
        abs_error = abs(values[0]-values[1])
        if abs_error > threshold:
            #extract file name of outlier to a list
            outlier_list[values[2]]=abs_error
    return dict(sorted(outlier_list.items(), key=lambda item: item[1]))

def plt_partial(real, pre, text):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real, y=pre, mode='markers', hovertext=text))
    fig.add_trace(go.Scatter(x=[i for i in range(-3,6)], y=[i for i in range(-3,6)], mode='lines', hovertext=text))
    fig.update_layout(     
                            font = dict(
                                        family='Times New Roman',color='black', size=22),
                  
                            plot_bgcolor = 'white',
                            xaxis = dict(range=[-3,5],
                                        title='DFT (eV)',
                                        showline=True,
                                        showgrid=True,
                                        gridcolor="white",
                                        showticklabels=True,
                                        linecolor='black',
                                        linewidth=2,
                                        ticks='outside',
                                        mirror=True,),

                            yaxis = dict(
                                        range=[-3,5],
                                        #title='<b>Bold</b> <i>animals</i>',
                                        title='ML (eV)',
                                        showline=True,
                                        showgrid=True,
                                        gridcolor="white",
                                        showticklabels=True,
                                        linecolor='black',
                                        linewidth=2,
                                        ticks='outside',
                                        mirror=True,)
                         )
    
    fig.show()

def plt_kpca(kernel_matrix, y, name_list=None):
    kpca = KernelPCA(kernel='precomputed', n_components=2)
    x_kpca = kpca.fit_transform(kernel_matrix)
    colors  = y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_kpca[:,0], y=x_kpca[:,1], mode='markers',hovertext=name_list,
                             marker=dict(
                                 size=10,
                                 color=colors,
                                 colorscale="Viridis",
                                 colorbar=dict(title="ads_energy")
                                        )
                             
                             ))
    fig.update_layout(     
                            font = dict(
                                        family='Times New Roman',color='black', size=22),
                  
                            plot_bgcolor = 'white',
                            xaxis = dict(
                                        title='PC1',
                                        showline=True,
                                        showgrid=False,
                                        gridcolor="white",
                                        showticklabels=False,
                                        linecolor='black',
                                        linewidth=2,
                                        ticks='outside',
                                        mirror=True,),

                            yaxis = dict(
                                        #title='<b>Bold</b> <i>animals</i>',
                                        title='PC2',
                                        showline=True,
                                        showgrid=False,
                                        gridcolor="white",
                                        showticklabels=False,
                                        linecolor='black',
                                        linewidth=2,
                                        ticks='outside',
                                        mirror=True,)
                         )
    
    fig.show()
    fig.write_html("Result/kpca.html")




def KF_validation(kernel_matrix, y, ML_method, name_list=None, 
                  n_split=5, shuffle=True, random_state=0
                  ):
    kf = KFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
    train_RMSEs,train_MAEs,test_RMSEs,test_MAEs,test_outliers =[],[],[],[],{}
    train_reals,train_pres,test_reals,test_pres,train_names,test_names = [],[],[],[],[],[]
    loop_index = 0
    for train_index, test_index in kf.split(kernel_matrix[0],y):
        loop_index += 1
        #get train_matrix and test_matrix
        train_matrix = kernel_matrix[train_index][:,train_index]
        test_matrix  = kernel_matrix[test_index][:,train_index]
        #y_train, y_test, name_train, name_test
        y_train, y_test = y[train_index], y[test_index]
        name_train, name_test = name_list[train_index], name_list[test_index] #outlier
        #train a model
        if ML_method['ml'] == 'gpr':
            ml_model = Train_gpr(train_matrix, y_train, **ML_method)
        if ML_method['ml'] == 'krr':
            ml_model = Train_krr(train_matrix, y_train, **ML_method)   
        #predict training and validation
        train_pre = ml_model.predict(train_matrix)
        test_pre  = ml_model.predict(test_matrix)
        test_outlier = Check_outlier(y_test,test_pre,name_test)
        # evaluate MAE and RMSE
        train_MAE = mean_absolute_error(train_pre, y_train)
        test_MAE  = mean_absolute_error(test_pre, y_test)
        train_RMSE = mean_squared_error(train_pre, y_train,squared=False)
        test_RMSE  = mean_squared_error(test_pre, y_test,squared=False)
        
        # append
        train_pres.extend(train_pre)
        train_reals.extend(y_train)
        test_pres.extend(test_pre)
        test_reals.extend(y_test)
        train_names.extend(list(map(lambda x: x + '-' + str(loop_index) , name_train)))    #name with additional loopindex of KF
        test_names.extend(list(map(lambda x: x + '-' + str(loop_index) , name_test)))
        
        train_MAEs.append(train_MAE)
        train_RMSEs.append(train_RMSE)
        test_MAEs.append(test_MAE)
        test_RMSEs.append(test_RMSE)
        test_outliers.update(test_outlier)
    
    KF_validation.train_pre  = train_pres
    KF_validation.train_real = train_reals
    KF_validation.test_pre = test_pres
    KF_validation.test_real = test_reals
    KF_validation.train_name = train_names
    KF_validation.test_name  = test_names
    
    KF_validation.avr_train_MAE = mean(train_MAEs)
    KF_validation.avr_test_MAE = mean(test_MAEs)
    KF_validation.avr_train_RMSE = mean(train_RMSEs)
    KF_validation.avr_test_RMSE = mean(test_RMSEs)
    
    return KF_validation.train_pre, KF_validation.train_real,\
           KF_validation.test_pre,  KF_validation.test_real,\
           KF_validation.train_name, KF_validation.test_name,\
           KF_validation.avr_train_MAE, KF_validation.avr_test_MAE,\
           KF_validation.avr_train_RMSE, KF_validation.avr_test_RMSE
    
    

# def KF_validation(kernel_matrix, y, ML_method, name_list=None, 
#                   n_split=5,shuffle=True,random_state=0
#                   ):
#     kf = KFold(n_splits=n_split,shuffle=shuffle,random_state=random_state)
#     train_RMSEs,train_MAEs,test_RMSEs,test_MAEs,test_outliers =[],[],[],[],{}
#     for train_index, test_index in kf.split(kernel_matrix[0],y):
#         #get train_matrix and test_matrix
#         train_matrix = kernel_matrix[train_index][:,train_index]
#         test_matrix  = kernel_matrix[test_index][:,train_index]
#         #y_train, y_test, name_train, name_test
#         y_train, y_test = y[train_index], y[test_index]
#         name_train, name_test = name_list[train_index], name_list[test_index] #outlier
#         #train a model
#         if ML_method == 'gpr':
#             ml_model = Train_gpr(train_matrix, y_train)
#         if ML_method == 'krr':
#             ml_model = Train_krr(train_matrix, y_train)   
#         #predict training and validation
#         train_pre = ml_model.predict(train_matrix)
#         test_pre  = ml_model.predict(test_matrix)
#         test_outlier = Check_outlier(y_test,test_pre,name_test)
#         # evaluate MAE and RMSE
#         train_MAE = mean_absolute_error(train_pre, y_train)
#         test_MAE  = mean_absolute_error(test_pre, y_test)
#         train_RMSE = mean_squared_error(train_pre, y_train,squared=False)
#         test_RMSE  = mean_squared_error(test_pre, y_test,squared=False)
        
#         # append
#         train_MAEs.append(train_MAE)
#         train_RMSEs.append(train_RMSE)
#         test_MAEs.append(test_MAE)
#         test_RMSEs.append(test_RMSE)
#         test_outliers.update(test_outlier)
        
#     avr_train_MAE = mean(train_MAEs)
#     avr_test_MAE = mean(test_MAEs)
#     avr_train_RMSE = mean(train_RMSEs)
#     avr_test_RMSE = mean(test_RMSEs)
#     return avr_train_MAE, avr_train_RMSE, avr_test_MAE, avr_test_RMSE, test_outliers


