import math

import numpy as np
from transform import to_cartesian, to_spherical

PI = np.pi


def to_spherical_slow(x):
    """
    non vectorized implementation
    """
    result = []
    for element in range(0, len(x)):
        r = 0
        for i in range(0, len(x[element])):
            r += x[element][i] * x[element][i]
        r = math.sqrt(r)
        ret = [r]
        for i in range(0, len(x[element]) - 2):
            temp = np.clip(x[element][i] / r, -1, 1)
            ret.append(math.acos(temp))
            diff = np.clip(r * r - x[element][i] * x[element][i], 0, None)
            r = np.clip(math.sqrt(diff), 0, None)
        if (x[element][-1] >= 0):
            fraction = np.clip(x[element][-2] / r, -1, 1)
            ret.append(math.acos(fraction))
        else:
            ret.append(2 * math.pi - math.acos(x[element][-2] / r))
        ret = np.array(ret)
        result += [ret]
    return np.array(result)


def to_cartesian_slow(x):
    """
    non vectorized implementation
    """
    result = []
    for element in range(0, len(x)):
        r = x[element][0]
        multi_sin = 1
        ret = []
        for i in range(1, len(x[element]) - 1):
            ret.append(r * multi_sin * math.cos(x[element][i]))
            multi_sin *= math.sin(x[element][i])
        ret.append(r * multi_sin * math.cos(x[element][-1]))
        ret.append(r * multi_sin * math.sin(x[element][-1]))
        ret = np.array(ret)
        result += [ret]
    return np.array(result)


def test_spherical(N, M):
    """
    test if vectorized version yields same result as slow implementation
    """
    c = 100 * np.random.rand(1)
    b = 10 * np.random.rand(1)
    a = c * np.random.rand(N, M) + b
    diff = np.abs(to_spherical(a) - to_spherical_slow(a))
    assert np.all(diff <= 1e-8)


def test_cartesian(N, M):
    """
    test if vectorized version yields same result as slow implementation
    """
    c = 100 * np.random.rand(1)
    b = 10 * np.random.rand(1)
    a = c * np.random.rand(N, M) + b
    diff = np.abs(to_cartesian(a) - to_cartesian_slow(a))
    assert np.all(diff <= 1e-12)


def test_combined_1(N, M):
    """
    test if cartesian->spherical->cartesian is identity
    """
    c = 100 * np.random.rand(1)
    b = 10 * np.random.rand(1)
    a = c * np.random.rand(N, M) + b
    diff = np.abs(to_cartesian(to_spherical(a)) - a)
    assert np.all(diff <= 1e-10)


def test_combined_2(N, M):
    """
    test if spherical->cartesian->spherical is identity

    note: the mapping to_spherical is not injective on all angles that one would randomly generate -> restrict to some angles
    """
    c = 1 * np.random.rand(1)
    b = 1 * np.random.rand(1)
    a = 10 * c * np.random.rand(N, M) + b
    a[..., 1:] = a[..., 1:] % (PI / 2) + 0.2
    diff = np.abs(to_spherical(to_cartesian(a)) - a)
    assert np.all(diff <= 1)


if __name__ == "__main__":

    for _ in range(100):
        M = np.random.randint(2, 20, 1)[0]
        N = 100
        test_spherical(N, M)
    print("Vectorization test passed for spherical coordinates.")

    for _ in range(100):
        M = np.random.randint(2, 20, 1)[0]
        N = 100
        test_cartesian(N, M)
    print("Vectorization test passed for cartesian coordinates.")

    for _ in range(100):
        M = np.random.randint(2, 20, 1)[0]
        N = 10000
        test_combined_1(N, M)
    print("Test 'cartesian->spherical->cartesian' passed.")

    for _ in range(100):
        M = np.random.randint(2, 20, 1)[0]
        N = 10000
        test_combined_2(N, M)
    print("Test 'spherical->cartesian->spherical' (restricted on valid angles) passed.")

    print("All tests passed.")
