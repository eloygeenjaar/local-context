# This code is adapted from: https://github.com/ilkhem/iVAE/blob/master/data/data.py

"""
Script for generating piece-wise stationary data.

Each component of the independent latents is comprised of `ns` segments, and each segment has different parameters.\
Each segment has `nps` data points 9measurements).

The latent components are then mixed by an MLP into observations (not necessarily of the same dimension.
It is possible to add noise to the observations
"""

import os

import numpy as np
import scipy
import torch
from scipy.stats import hypsecant
from sklearn.model_selection import train_test_split


def to_one_hot(x, m=None):
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def lrelu(x, neg_slope):
    """
    Leaky ReLU activation function
    @param x: input array
    @param neg_slope: slope for negative values
    @return:
        out: output rectified array
    """

    def _lrelu_1d(_x, _neg_slope):
        """
        one dimensional implementation of leaky ReLU
        """
        if _x > 0:
            return _x
        else:
            return _x * _neg_slope

    leaky1d = np.vectorize(_lrelu_1d)
    assert neg_slope > 0  # must be positive
    return leaky1d(x, neg_slope)


def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input array
    @return:
        out: output array
    """
    return 1 / (1 + np.exp(-x))


def generate_mixing_matrix(d_sources: int, d_data=None, lin_type='uniform', cond_threshold=25, n_iter_4_cond=None,
                           dtype=np.float32):
    """
    Generate square linear mixing matrix
    @param d_sources: dimension of the latent sources
    @param d_data: dimension of the mixed data
    @param lin_type: specifies the type of matrix entries; either `uniform` or `orthogonal`.
    @param cond_threshold: higher bound on the condition number of the matrix to ensure well-conditioned problem
    @param n_iter_4_cond: or instead, number of iteration to compute condition threshold of the mixing matrix.
        cond_threshold is ignored in this case/
    @param dtype: data type for data
    @return:
        A: mixing matrix
    @rtype: np.ndarray
    """

    def _gen_matrix(ds, dd, dtype):
        A = (np.random.uniform(0, 2, (ds, dd)) - 1).astype(dtype)
        for i in range(dd):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        return A

    if d_data is None:
        d_data = d_sources

    if lin_type == 'orthogonal':
        A = (np.linalg.qr(np.random.uniform(-1, 1, (d_sources, d_data)))[0]).astype(dtype)

    elif lin_type == 'uniform':
        if n_iter_4_cond is None:
            cond_thresh = cond_threshold
        else:
            cond_list = []
            for _ in range(int(n_iter_4_cond)):
                A = np.random.uniform(-1, 1, (d_sources, d_data)).astype(dtype)
                for i in range(d_data):
                    A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
                cond_list.append(np.linalg.cond(A))

            cond_thresh = np.percentile(cond_list, 25)  # only accept those below 25% percentile

        gen_mat = _gen_matrix
        A = gen_mat(d_sources, d_data, dtype)
        while np.linalg.cond(A) > cond_thresh:
            A = gen_mat(d_sources, d_data, dtype)

    else:
        raise ValueError('incorrect method')
    return A


def generate_nonstationary_sources(n_per_seg: int, n_seg: int, d: int, prior='gauss', var_bounds=[0.5, 3],
                                   dtype=np.float32, uncentered=False):
    """
    Generate source signal following a TCL distribution. Within each segment, sources are independent.
    The distribution withing each segment is given by the keyword `dist`
    @param n_per_seg: number of points per segment
    @param n_seg: number of segments
    @param d: dimension of the sources same as data
    @param prior: distribution of the sources. can be `lap` for Laplace , `hs` for Hypersecant or `gauss` for Gaussian
    @param var_bounds: optional, upper and lower bounds for the modulation parameter
    @param dtype: data type for data
    @param bool uncentered: True to generate uncentered data
    @return:
        sources: output source array of shape (n, d)
        labels: label for each point; the label is the component
        m: mean of each component
        L: modulation parameter of each component
    @rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    var_lb = var_bounds[0]
    var_ub = var_bounds[1]
    n = n_per_seg * n_seg

    print(var_lb, var_ub, n_seg, d)
    L = np.random.uniform(var_lb, var_ub, size=(n_seg, d))
    if uncentered:
        m = np.random.uniform(-5, 5, size=(n_seg, d))
    else:
        m = np.zeros((n_seg, d))

    labels = np.zeros(n, dtype=dtype)
    if prior == 'lap':
        sources = np.random.laplace(0, 1 / np.sqrt(2), (n, d)).astype(dtype)
    elif prior == 'hs':
        sources = scipy.stats.hypsecant.rvs(0, 1, (n, d)).astype(dtype)
    elif prior == 'gauss':
        sources = np.random.randn(n, d).astype(dtype)
    else:
        raise ValueError('incorrect dist')

    for seg in range(n_seg):
        segID = range(n_per_seg * seg, n_per_seg * (seg + 1))
        sources[segID] *= L[seg]
        sources[segID] += m[seg]
        labels[segID] = seg

    return sources, labels, m, L


def generate_data(n_per_seg, n_seg, d_sources, d_data=None, n_layers=3, prior='gauss', activation='lrelu',
                  seed=10, slope=.1, var_bounds=[0.5, 3], lin_type='uniform', n_iter_4_cond=1e4,
                  dtype=np.float32, noisy=0, uncentered=False, discrete=False,
                  repeat_linearity=True):
    """
    Generate artificial data with arbitrary mixing
    @param int n_per_seg: number of observations per segment
    @param int n_seg: number of segments
    @param int d_sources: dimension of the latent sources
    @param int or None d_data: dimension of the data
    @param int n_layers: number of layers in the mixing MLP
    @param str activation: activation function for the mixing MLP; can be `none, `lrelu`, `xtanh` or `sigmoid`
    @param str prior: prior distribution of the sources; can be `lap` for Laplace or `hs` for Hypersecant
    @param int batch_size: batch size if data is to be returned as batches. 0 for a single batch of size n
    @param int seed: random seed
    @param var_bounds: upper and lower bounds for the modulation parameter
    @param float slope: slope parameter for `lrelu` or `xtanh`
    @param str lin_type: specifies the type of matrix entries; can be `uniform` or `orthogonal`
    @param int n_iter_4_cond: number of iteration to compute condition threshold of the mixing matrix
    @param dtype: data type for data
    @param float noisy: if non-zero, controls the level of noise added to observations
    @param bool uncentered: True to generate uncentered data
    @param bool one_hot_labels: if True, transform labels into one-hot vectors

    @return:
        tuple of batches of generated (sources, data, auxiliary variables, mean, variance)
    @rtype: tuple

    """
    if seed is not None:
        np.random.seed(seed)

    if d_data is None:
        d_data = d_sources

    # sources
    S, U, M, L = generate_nonstationary_sources(n_per_seg, n_seg, d_sources, prior=prior,
                                                var_bounds=var_bounds, dtype=dtype,
                                                uncentered=uncentered)
    n = n_per_seg * n_seg

    # non linearity
    if activation == 'lrelu':
        act_f = lambda x: lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'xtanh':
        act_f = lambda x: np.tanh(x) + slope * x
    elif activation == 'none':
        act_f = lambda x: x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))

    # Mixing time!

    if not repeat_linearity:
        X = S.copy()
        for nl in range(n_layers):
            A = generate_mixing_matrix(X.shape[1], d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
            if nl == n_layers - 1:
                X = np.dot(X, A)
            else:
                X = act_f(np.dot(X, A))

    else:
        assert n_layers > 1  # suppose we always have at least 2 layers. The last layer doesn't have a non-linearity
        A = generate_mixing_matrix(d_sources, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
        X = act_f(np.dot(S, A))
        if d_sources != d_data:
            B = generate_mixing_matrix(d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
        else:
            B = A
        for nl in range(1, n_layers):
            if nl == n_layers - 1:
                X = np.dot(X, B)
            else:
                X = act_f(np.dot(X, B))

    # add noise:
    if noisy:
        X += noisy * np.random.randn(*X.shape)

    if discrete:
        X = np.random.binomial(1, sigmoid(X))

    U = to_one_hot([U], m=n_seg)[0]
    return S, X, U, M, L


def save_data(path, **kwargs):
    S, X, U, M, L = generate_data(**kwargs)
    print('Creating dataset {} ...'.format(path))
    dir_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs('/'.join(path.split('/')[:-1]))
    np.savez_compressed(path, s=S, x=X, u=U, m=M, L=L)
    print(' ... done')


def create_if_not_exist_dataset(root='data/', nps=1000, ns=40, dl=2, dd=4, nl=3, s=1, p='gauss', a='xtanh',
                                uncentered=False, noisy=False):
    """
    Create a dataset if it doesn't exist.
    This is useful as a setup step when running multiple jobs in parallel, to avoid having many scripts attempting
    to create the dataset when non-existent.
    This is called in `cmd_utils.create_dataset_before`
    """

    path_to_dataset = root + 'tcl_' + '_'.join(
        [str(nps), str(ns), str(dl), str(dd), str(nl), str(s), p, a])
    if uncentered:
        path_to_dataset += '_u'
    if noisy:
        path_to_dataset += '_n'
    path_to_dataset += '.npz'

    if not os.path.exists(path_to_dataset) or s is None:
        kwargs = {"n_per_seg": nps, "n_seg": ns, "d_sources": dl, "d_data": dd, "n_layers": nl, "prior": p,
                  "activation": a, "seed": s, "uncentered": uncentered, "noisy": noisy}
        save_data(path_to_dataset, **kwargs)
    return path_to_dataset

def index_and_convert_data(X, S, U, ix):
    return (torch.from_numpy(X[:, ix]).float(),
            torch.from_numpy(S[:, ix]).float(),
            torch.from_numpy(U[:, ix]).float())

def split_dataset(data, data_type, shape, seed):
    num_samples, num_segments, input_size, latent_dim = shape
    X = data['x']
    S = data['s']
    U = data['u']
    X = X.reshape(num_segments, num_samples, input_size)
    S = S.reshape(num_segments, num_samples, latent_dim)
    U = U.reshape(num_segments, num_samples, num_segments)
    ix = np.arange(num_samples)
    train_val, test_ix = train_test_split(ix, test_size=0.2, random_state=seed)
    train_ix, valid_ix = train_test_split(train_val, test_size=0.1, random_state=seed)
    if data_type == 'tr':
        X, S, U = index_and_convert_data(X, S, U, train_ix)
    elif data_type == 'va':
        X, S, U = index_and_convert_data(X, S, U, valid_ix)
    elif data_type =='te':
        X, S, U = index_and_convert_data(X, S, U, test_ix)
    else:
        X, S, U = None, None, None

    X = torch.reshape(X, (-1, input_size))
    S = torch.reshape(S, (-1, latent_dim))
    U = torch.reshape(U, (-1, num_segments))
    return X, S, U
