from GPR import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.kernel_ridge import KernelRidge
def Train_gpr(features, target, **kwargs):
    """
    kernel=None, alpha=1e-10, normalize_y=True
    alpha=1e-10
    docstring
    """
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 1e-10
    normalize_y = kwargs['normalize_y'] if 'normalize_y' in kwargs else False
    
    gpr = GaussianProcessRegressor(kernel='precomputed', alpha=alpha, normalize_y=normalize_y)
    gpr.fit(features,target)
    return gpr

def Train_krr(features,target,alpha=0.000001,kernel=None):
    """
    docstring
    """
    krr = KernelRidge(kernel='precomputed',alpha=alpha)
    krr.fit(features,target)
    return krr


if __name__ == "__main__":
   pass
    