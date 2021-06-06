import numpy as np
import matplotlib.pyplot as plt


# Constant
C = 3           # No. Cluster
K = 42          # No. data
M = 2           # Fuzzy controller
F = 1           # Flag : 1 for Fuzzy, 0 for Hard clustring

def compute_J(d, U):
    """
    Compute Error function : 
    Hard c-mean error : SUM(U * D^2)
    Fuzzy c-means error : SUM(U^M * D^2)
    Parameters
    ------
    d : K * C array
        distance of each data from C
    U : C * K array
        Membership values of data for each cluster

    Returns
    ------
    Calculated error
    """   

    if F:
        U = U ** M
    d = d ** 2
    J = np.sum(np.dot(U, d))
    return J


def distance(X, centers):
    """
    Compute distance of each data from centers
    Parameters
    ------
    X : K * 2 array
        Our data
    centers : C * 2 array
        Center of each cluster

    Returns
    ------
    dis : K * C array
        Distance of data from each center
    """
    dis = []
    for i in range(K):
        tmp = []
        x = X[i]
        for j in range(C):
            c = centers[j]
            d2 = (x[0] - c[0])**2 + (x[1]-c[1])**2
            d = np.sqrt(d2)
            tmp.append(np.round(d, 4))
        dis.append(tmp)
    dis = np.array(dis)
    return dis


def center(X, U):
    """
    Compute Center Hard c means: Sum(X * U) / SUM(U)
    Compute Center fuzzy c means: Sum(X^2 * U) / SUM(U)

    Parameters
    ------
    X : K * 2 array
        
    U : C * K array
        Membership values of data for each cluster

    Returns
    ------
    centers : C * 2 array
    """
    centers = []
    p = 1
    if F:
        p = M

    for i in range(C):
        tmp = []
        c = U[i:i+1,:] ** p
        x = X[:,0:1]
        y = X[:,1:2]

        cx = np.sum(c.dot(x)) / np.sum(c)
        cy = np.sum(c.dot(y)) / np.sum(c)

        tmp.append(np.round(cx, 3))
        tmp.append(np.round(cy, 3))
        centers.append(tmp)
    centers = np.array(centers)
    return centers


def generate_U():
    """
    Compute random U array

    Returns
    ------
    centers : C * K random array
    """
    zero_mat = np.zeros((C, K))
    for j in range(K):
        i = np.random.randint(0,3)
        zero_mat[i][j] = 1

    return zero_mat



def update_fuzzy(d, U):
    """
    Update membership values for fuzzy c-means --> U

    Parameters
    ------
    d : K * C array
        distance of each data from C
    U : C * K array
        Membership values of data for each cluster

    Returns
    ------
    centers : Updated U --> C * K array
    """
    compute_J(d.copy(), U.copy())
    for i in range(K):
        # each data
        u = U[:,i]
        dis = d[i]    #dik
        flag = False
        for j in range(C):
            x = dis[j]
            u_ik = []
            for k in range(C):
            
                if dis[k] != 0:
                    tmp = x / dis[k]
                    tmp = tmp ** (2/(M-1))
                    u_ik.append(tmp)
                else:
                    u[dis != dis[k]] = 0
                    flag = True
                    break
            if flag:
                break
            s = np.sum(u_ik)
            s = 1 / s
            u[j] = np.round(s, 3)
      
        u = u.reshape(len(u), -1)
        U[:,i] = u[:,0]

    return U


def main_fuzz(X):
    """
    main function for fuzzy c-means clustring

    Parameters
    ------
    X : K * 2 array
    """
    myU = []
    U = generate_U()
    myU.append(U)

    y = []
    x = []
    for i in range(1000):
        centers = center(X,U)
        dis = distance(X, centers)

        j = compute_J(dis.copy(), U.copy())

        y.append(j)
        x.append(i)
        
        U = update_fuzzy(dis, U.copy())
        myU.append(U)


        # Stop condition
        tmp = myU[i+1] - myU[i]
        m = np.amax(tmp)
        if m < 0.01:
            break

def update_hard(d, U):
    """
    Update membership values for hard c-means --> U

    Parameters
    ------
    d : K * C array
        distance of each data from C
    U : C * K array
        Membership values of data for each cluster

    Returns
    ------
    centers : Updated U --> C * K array
    """
    for i in range(K):
        dx = d[i]
        j = np.where(dx == min(dx))[0][0]

        tmp = np.zeros((C,1))
        U[:,i] = tmp[:,0]
        U[:,i][j] = 1
    return U

def plot(X, U):
    """
    Plotting clustered data
    """
    x = X[:, 0:1]
    y = X[:,1:2]

    for i in range(C):
        u = U[i]
        plt.scatter(x[u == 1], y[u == 1])
    plt.show()

def main_hard(X):

    U = generate_U()

    for i in range(10):
        centers = center(X,U)
        dis = distance(X, centers)
        compute_J(dis.copy(), U.copy())
        U = update_hard(dis, U)
    plot(X, U)


if __name__ == "__main__":
    data = np.array([[0, -2],[0, -1],[0, 1],[0, 2],[0, -5],[0, -6],[1, -1],[1, 0],
            [1, 2],[1, 3],[1, -5],[1, -6],[2, 0],[2, 1],[-2, -1],[-2, 0],
            [2, -5],[2, -6],[-1, -3],[-1, -2],[-1, 0],[-1, 1],[-1, -5],
            [-1, -6],[3, 1],[3, 2],[-3, -2],[-3, -1],[-3, -5],[-3, -6],
            [4, 2],[4, 3],[-4, 3],[-4, -2],[-4, -5],[-4, -6],[-2, -4],
            [-2, -3],[2, 3],[2, 4],[-6, -5],[-5, -6]])

    if F:
        main_fuzz(data)
    else:
        main_hard(data)
    
