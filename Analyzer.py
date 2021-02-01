
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from ML_learning import Train_gpr, Train_krr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statistics import mean

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
    

def KF_validation(kernel_matrix, y, ML_method, name_list=None, 
                  n_split=5,shuffle=True,random_state=0
                  ):
    kf = KFold(n_splits=n_split,shuffle=shuffle,random_state=random_state)
    train_RMSEs,train_MAEs,test_RMSEs,test_MAEs,test_outliers =[],[],[],[],{}
    for train_index, test_index in kf.split(kernel_matrix[0],y):
        #get train_matrix and test_matrix
        train_matrix = kernel_matrix[train_index][:,train_index]
        test_matrix  = kernel_matrix[test_index][:,train_index]
        #y_train, y_test, name_train, name_test
        y_train, y_test = y[train_index], y[test_index]
        name_train, name_test = name_list[train_index], name_list[test_index] #outlier
        #train a model
        if ML_method == 'gpr':
            ml_model = Train_gpr(train_matrix, y_train)
        if ML_method == 'krr':
            ml_model = Train_krr(train_matrix, y_train)   
        #predict training and validation
        train_pre = ml_model.predict(train_matrix)
        test_pre  = ml_model.predict(test_matrix)
        test_outlier = Check_outlier(y_test,test_pre,name_test)
        #test_outlier= Check_outlier(y_test,test_pre,name_test)
        # evaluate MAE and RMSE
        train_MAE = mean_absolute_error(train_pre, y_train)
        test_MAE  = mean_absolute_error(test_pre, y_test)
        train_RMSE = mean_squared_error(train_pre, y_train,squared=False)
        test_RMSE  = mean_squared_error(test_pre, y_test,squared=False)
        # append
        train_MAEs.append(train_MAE)
        train_RMSEs.append(train_RMSE)
        test_MAEs.append(test_MAE)
        test_RMSEs.append(test_RMSE)
        test_outliers.update(test_outlier)
        
    avr_train_MAE = mean(train_MAEs)
    avr_test_MAE = mean(test_MAEs)
    avr_train_RMSE = mean(train_RMSEs)
    avr_test_RMSE = mean(test_RMSEs)
    return avr_train_MAE, avr_train_RMSE, avr_test_MAE, avr_test_RMSE,test_outliers


