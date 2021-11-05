import numpy as numpy


def pca(PCA):
    data = np.array([[1, 2, 3], [1, 2, 5], [0, 0, 2], [0, 1, 1], [2, 2, 2]])
    pca = PCA(data)

    # __init__ checks
    assert pca.N == 5 and pca.p == 3
    assert pca.mu.shape == (pca.p,), "Wrong dimensionality of mean data vector"
    assert pca.eig_vals.shape == (pca.p,), "Incorrect number of eigenvalues, check the covariance matrix dimensions (p, p)!"
    assert np.allclose(pca.mu, [0.8, 1.4, 2.6]), "Incorrect calculation of mean data vector"
    assert np.allclose(pca.eig_vals, [2.75325386, 0.91818597, 0.12856017]), "Incorrect calculation of eigenvalues"
    vecs = np.array([[-0.28648964, -0.67146482, -0.68341692],
                     [-0.40000173, -0.56434614, 0.72215791],
                     [-0.87058733, 0.48025871, -0.10690776]])
    signs = np.ones((pca.p,))  # Take care of possibly different signs for eigenvectors
    valid = True
    for i in range(pca.p):
    if np.allclose(pca.eig_vecs[:,i], -vecs[:,i]):
        signs[i] = -1
    elif not np.allclose(pca.eig_vecs[:,i], vecs[:,i]):
        valid = False
    assert valid, "Incorrect calculation of eigenvectors"

    # project and reconstruct checks
    for q in range(1, pca.p):
    assert pca.E(q).shape == (pca.p, q), "Incorrect dimensionality of projection matrix E"
    z = pca.project(data, q)
    assert z.shape == (pca.N, q), "Wrong dimensionality of projected vector"
    x = pca.reconstruct(z)
    assert x.shape == (pca.N, pca.p), "Wrong dimensionality of projected vector"
    z = pca.project(data, pca.p)
    zc = [[-0.6455339, -0.28079716, 0.25384826],
          [-2.38670856, 0.67972026, 0.04003273],
          [1.31154653, 1.03910123, -0.40014289],
          [1.78213214, -0.00550363, 0.42892279],
          [-0.06143621, -1.43252069, -0.32266089]]
    assert np.allclose(z*signs, zc), "Incorrect projection of data"
    x = pca.reconstruct(z)
    assert np.allclose(x, data), "Incorrect reconstruction of projected data"

    # mse checks
    for q in range(1, pca.p):
    assert isinstance(pca.mse(data, q), float), "Incorrect return value of mse function, this should return a scalar float."
    assert np.isclose(pca.mse(data, 2), 0.03428271323938805)

    print("Your implementation works correctly!")
