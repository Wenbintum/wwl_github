from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.kernel_ridge import KernelRidge
def Train_gpr(features,target,kernel=None):
    """
    docstring
    """
    pass
    gpr = GaussianProcessRegressor(kernel=PairwiseKernel(metric='precomputed'))
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
    