import numpy as np
import scipy.sparse as spr
from sklearn.neighbors import KDTree

def DataNorm(X):

    m = np.mean(X, 0)
    std = np.std(X, 0, ddof=1)

    return (X - m)/std

def SparseCoordinate(M):

    L = M.shape[0]
    C = M.shape[1]

    A = np.zeros((L,L))

    X = np.repeat(np.array(range(L)).reshape(-1,1), C-1, axis=1).reshape(-1)
    Mt = M[:, 1:].reshape(-1)
    A[X, Mt] = 1.
    A[Mt, X] = 1.

    ind = np.where(A==1.)
    index_x = ind[0].reshape(-1,1)
    index_y = ind[1].reshape(-1,1)

    return index_x, index_y


def optgamma(xt, x, Q, c):
    d = xt - x
    a = d.T @ Q @ d
    b = 2 * x.T @ Q @ d + 2 * c @ d

    if np.abs(a)<1e-7:
        mv=np.inf
        if (a<0) & (b>0):
            m=np.inf
        elif (a<0) &(b<0):
            m=-np.inf
        elif (a>0) &(b>0):
            m=-np.inf
        else:
            m=np.inf

    else:
        m = -b/(2*a)
        mv = a*m**2 + b*m

    s = np.array([mv, a + b, 0], dtype=object)
    I = np.argsort(s)

    if I[-1]==0:
        if m > 1:
            gamma = 1
        elif m < 0:
            gamma = 0
        else:
            gamma = m
    elif I[-1]==1:
        gamma = 1
    else:
        gamma = 0

    return gamma


def FrankWolfe(K1, K2, K3, x):

    N = x.shape[0]
    vecone = np.ones((N,1),dtype=np.float64)
    Q = K1 - 2*K2 + K3

    c = vecone.T @ (K2-K3)
    eps = 1e-3

    iter = 0
    t = 10
    gamma = 1
    xt = np.zeros((N,1))

    while (t > eps) & (iter < 100) & (gamma != 0):
        x0 = x
        G = Q @ x0 + c.T
        xt[G>=0] = 1
        xt[G<0] = 0
        if (xt==x).all():
            break
        else:
            gamma = optgamma(xt,x,Q,c)
            x = x + gamma*(xt - x)
            iter = iter + 1
            t = np.linalg.norm(x - x0)
    x = Q @ x + c.T
    x[x>=0] = 1
    x[x<0] = 0
    obj = x.T @ Q @ x + c @ x
    return x, obj



def MCDM_filter(X, Y,K,lamda):

    _, unq = np.unique(X, axis=0, return_index=True)
    X = X[unq]
    Y=Y[unq]
    N = X.shape[0]

    K0 = min(int(N/20),50)
    lbd0 = 1.0
    d = 3.05
    s = 0.2
    kdTreeX = KDTree(X)
    _, Neighbor = kdTreeX.query(X, K0+1)

    IS, JS = SparseCoordinate(Neighbor)
 
    Vector = Y - X
    VECn = DataNorm(Vector)
    Xn = DataNorm(X)

    IS = IS.squeeze()
    JS = JS.squeeze()
    p = np.exp(-lbd0*np.linalg.norm(VECn[IS]-VECn[JS], axis=1)*np.sqrt(1/((Xn[IS,0]-Xn[JS,0])**2+s)+1/((Xn[IS,1]-Xn[JS,1])**2+s)))
    v1 = np.log(p)
    v2 = np.log((1-p)/d)
    v3 = np.log(1-p-2*(1-p)/d)
    v1[v1==np.nan] = -100
    v2[v2==np.nan] = -100
    v3[v3==np.nan] = -100

    K1 = spr.csr_matrix((v1.reshape(-1),(IS.reshape(-1),JS.reshape(-1))),shape=(N,N),dtype=np.float)
    K2 = spr.csr_matrix((v2.reshape(-1),(IS.reshape(-1),JS.reshape(-1))),shape=(N,N),dtype=np.float)
    K3 = spr.csr_matrix((v3.reshape(-1),(IS.reshape(-1),JS.reshape(-1))),shape=(N,N),dtype=np.float)
    xl, obj = FrankWolfe(K1,K2,K3,np.ones((N,1)))

    idx = np.where(xl>0.5)[0]


    if len(idx) > K:

        X2 = X[idx]
        kdTreeX2 = KDTree(X2)
        _,Neighbor2 = kdTreeX2.query(X, K+1)
        Neighbor2 = idx[Neighbor2]

        IS, JS = SparseCoordinate(Neighbor2)

        IS = IS.squeeze()
        JS = JS.squeeze()
        p = np.exp(-lamda*np.linalg.norm(VECn[IS]-VECn[JS], axis=1)*np.sqrt(1/((Xn[IS,0]-Xn[JS,0])**2+s)+1/((Xn[IS,1]-Xn[JS,1])**2+s)))
        v1l = np.log(p)
        v2l = np.log((1-p)/d)
        v3l = np.log(1-p-2*(1-p)/d)
        v1l[v1l==np.nan] = -100
        v2l[v2l==np.nan] = -100
        v3l[v3l==np.nan] = -100

        K1l = spr.csr_matrix((v1l.reshape(-1),(IS.reshape(-1),JS.reshape(-1))),shape=(N,N),dtype=np.float)
        K2l = spr.csr_matrix((v2l.reshape(-1),(IS.reshape(-1),JS.reshape(-1))),shape=(N,N),dtype=np.float)
        K3l = spr.csr_matrix((v3l.reshape(-1),(IS.reshape(-1),JS.reshape(-1))),shape=(N,N),dtype=np.float)
        x2, obj2 = FrankWolfe(K1l, K2l, K3l, np.ones((N,1)))

        idx = np.where(x2>0.5)[0]


    return idx