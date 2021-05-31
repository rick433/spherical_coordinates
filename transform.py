import numpy as np
from numpy import arccos, cos, sin, sqrt, clip, stack, zeros

PI = np.pi


def to_spherical(x):
    """
    Transforms a batch of M-dimensional cartesian data points into spherical coordinates

    Careful: Some points on the n-sphere can be expressed given different angles. 
             E.g. on the unit 2-sphere the north pole can be given by (r=1,theta=pi/2,phi) 
             for any phi. Check your data to avoid inconsitencies. 

    Parameters:

    x: np.array of shape [N,M] with N the number of data points and M the dimension

    Returns:

    np.array of shape [N,M] 
    """
    r = sqrt((x ** 2).sum(1))
    N = x.shape[0]
    M = x.shape[1]
    ret = [r]
    for i in range(M - 2):
        temp = x[..., i] / r
        ret.append(arccos(clip(temp, -1, 1)))
        r = sqrt(clip(r ** 2 - x[..., i] ** 2, 0, None))
    last_phis = zeros(N)
    ## x[..,,-2] > 0 
    mask = x[..., -1] > 0
    last_phis[mask] = arccos(clip(x[mask, -2] / r[mask], -1, 1))
    ## x[..,,-2] < 0 
    mask = x[..., -1] < 0
    last_phis[mask] = 2 * PI - arccos(clip(x[mask, -2] / r[mask], -1, 1))
    ## x[..,,-2] = 0 
    mask = x[..., -1] == 0
    last_phis[mask] = 3 / 2 * PI
    ret.append(last_phis)
    return stack(ret).transpose()


def to_cartesian(x):
    """
    Transforms a batch of M-dimensional data given in spherical coordinates into cartesian coordinates

    Parameters:

    x: np.array of shape [N,M] with N the number of data points and M the dimension

    Returns:

    np.array of shape [N,M] 

    """
    result = []
    r = x[..., 0]
    temp = r
    for i in range(1, x.shape[1]):
        x_i = temp * cos(x[..., i])
        temp = temp * sin(x[..., i])
        result.append(x_i)
    result.append(temp)
    return stack(result).transpose()
