import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import ortho_group
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn import metrics
import scipy as sp
from hilbertcurve.hilbertcurve import HilbertCurve
import torch
import torch.nn.functional as F
import numpy as np
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

def rec_loss_function(recon_x, x, rec_type):
    # if rec_type == 'BCE':
    #     reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    if rec_type == 'MSE':
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    elif rec_type == 'MAE':
        reconstruction_loss = F.l1_loss(recon_x, x, reduction='sum')
    else:
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum') + F.l1_loss(recon_x, x, reduction='sum')
    return reconstruction_loss

def trans(X, k=5):
    # batch, channels, height, width = X.shape
    # X = X.view(batch, -1)
    Xs = X - X.min(1, keepdim=True)[0]
    Xs = Xs / Xs.max(1, keepdim=True)[0]
    Xs = (Xs * (2**k-1)).long()
    Xs[Xs==(2**k)] = 2**k-1
    return Xs

def hilbert_order(X, p=5):
    n, d = X.shape
    
    hilbert_curve = HilbertCurve(p, d)
    Xi = trans(X, p)
    Xdistances = torch.tensor(hilbert_curve.distances_from_points(Xi.cpu().numpy()), dtype=torch.float32).to(X.device) / (2**(d*p) - 1)
    Xr = torch.argsort(Xdistances)
    
    return Xr

## equal weights HCP
def HCP(X, Y, hc='Hk', k=5):

    """
    Implement Hilbert curve projection distance (p=2) for equal sample size and equal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (n, d), samples in the target domain
        hc : string, impementation of the Hilbert curve, either 'Hm' or 'Hk'. 'Hm' is based on recursive sort, and 'Hk' is based on Hilbert indices. For large n, Hm is recommended.
        k : int, order of Hilbert curve when hc='Hk'

    Returns
    ----------
        float, Hilbert curve projection distance (p=2)
    """

    n,d = X.shape
    # batch, channels, height, width = X.shape
    # d = channels*height*width
    assert (hc == 'Hm' or hc == 'Hk'), \
        "Hilbert curve should only be Hm or Hk"

    if hc=='Hm':
        # recursive sort
        # Xr = base.hilbert_order(X)
        # Yr = base.hilbert_order(Y)
        raise NotImplementedError("Recursive sort Hm is based on base.cpp and is not implemented yet")
    else:
        # Hilbert indices based sort
        p=k
        hilbert_curve = HilbertCurve(p, d)

        Xi = trans(X, k)
        Yi = trans(Y, k)
        Xdistances = torch.tensor(hilbert_curve.distances_from_points(Xi))/(2**(d*p) - 1)
        Xr = torch.argsort(Xdistances)
        Ydistances = torch.tensor(hilbert_curve.distances_from_points(Yi))/(2**(d*p) - 1)
        Yr = torch.argsort(Ydistances)
    res = F.mse_loss(X[Xr,:], Y[Yr,:], reduction='sum')
    # res = torch.sum(((X[Xr,:]-Y[Yr,:])**2))/

    return torch.sqrt(res)





#######################################################
## Only for equal weights
#######################################################
def rand_projections(embedding_dim, num_samples=50):
    
    projections = [w / np.sqrt((w**2).sum())
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    projections = torch.from_numpy(projections).to(torch.float)
    return projections

## equal weights IPRHCP
def IPRHCP(X, Y, q=2, nslice=50, direction='ortho'):

    """
    Implement integral projection robust Hilbert curve projection distance (p=2) for equal sample size and equal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (n, d), samples in the target domain
        q : int, dimension of subspace
        nslice : int, number of slices 
        direction : string, kind of the sliced direction, either 'ortho' or 'random'. 'ortho' means orthogonal directions, and 'random' means directions are independent and uniform to sphere.

    Returns
    ----------
        float, integral projection robust Hilbert curve projection distance (p=2) for equal sample size and equal weights
    """

    assert (direction == 'ortho' or direction == 'random'), \
        "Direction should only be orthogonal or random"
    
    d = X.shape[1]
    res = torch.tensor(0.0, device=X.device)

    if direction=='ortho':
        ## to generate less orthogonal matrix(here we need d/q is integer)
        k = int(q*nslice/d)+1
        projM = torch.zeros((d,d*k), device=X.device)
        for j in range(k):
            projM[:,(j*d):(j*d+d)] = torch.tensor(ortho_group.rvs(dim=d), dtype=torch.float32)

        for i in range(nslice):
            proj = projM[:,(i*q):(i*q+q)]
            # print(f"projM device: {projM.device}")
            # print(f"X device: {X.device}")
            # print(f"proj device: {proj.device}")
            Xi = X@proj
            Yi = Y@proj
            res += HCP(Xi, Yi)**2

    else:
        ## random directions may be faster
        proj = rand_projections(d, nslice*q)
        Xp = X@proj.T
        Yp = Y@proj.T

        for i in range(nslice):
            Xi = Xp[:,(i*q):(i*q+q)]
            Yi = Yp[:,(i*q):(i*q+q)]
            res += HCP(Xi, Yi)**2

    
    return torch.sqrt(res/nslice)



#######################################################
## equal weights IPRHCP
def PRHCP(X, Y, q=2, niter=10):

    """
    Implement projection robust Hilbert curve projection distance (p=2) for equal sample size and equal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (n, d), samples in the target domain
        q : int, dimension of subspace
        niter : int, maximum number of iterations 

    Returns
    ----------
        float, projection robust Hilbert curve projection distance (p=2) for equal sample size and equal weights
    """
    
    n, d = X.shape
    t = 0
    tau = 1
    i = 0
    Omega = np.eye(d)
    device = X.device
    X = X.cpu()
    Y = Y.cpu()
    while i<niter:
        if i==0:
            Xr = hilbert_order(X)
            Yr = hilbert_order(Y)
            delta = (X[Xr,:]-Y[Yr,:]).cpu()
        else:
            Xu = X@U
            Yu = Y@U
            Xr = hilbert_order(Xu)
            Yr = hilbert_order(Yu)
            delta = (X[Xr,:]-Y[Yr,:]).cpu()
        
        disp = np.concatenate([delta, -delta])
        pca = PCA(n_components = q, random_state = 1)
        pca.fit(disp)
        U = pca.components_.T
        Omega = (1-tau)*Omega + tau*U@U.T
        eigenvalues, eigenvectors = sp.linalg.eigh(Omega, eigvals=(d-q,d-1))
        U = eigenvectors
        t += 1
        tau = 2/(2+t)
        i += 1
    
    U = torch.tensor(U, dtype=torch.float32)
    X.to(device)
    distance = HCP(X@U,Y@U)

    return distance

def hcp_loss_fn(X, hcp_type = "PRHCP"):
    z = torch.randn(X.size()).to(X.device)
    if hcp_type == "PRHCP":
        return PRHCP(X, z)
    elif hcp_type == "IPRHCP":
        return IPRHCP(X, z)
    elif hcp_type == "HCP":
        return HCP(X, z)
    else:
        raise NotImplementedError(f"HCP type: {hcp_type} not implemented")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    X = torch.randn(100, 10).to(device)
    Y = torch.randn(100, 10).to(device)
    print(f"HCP-distance: {HCP(X, Y)}")
    print(f"IPHCP-distance {IPRHCP(X, Y)}")
    print(f"PRHCP-distance {PRHCP(X, Y)}")
    vae= AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    print(vae.parameters())
