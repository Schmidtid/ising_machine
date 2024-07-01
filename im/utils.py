import numpy as np

__all__ = ['ising_qubo', 'qubo_ising', 'load_graph']

def ising_qubo(J: np.ndarray, h:np.ndarray=None):
    """Convert Ising Hamiltonian to QUBO with \
    offset == J.sum() - h.sum()
    
    Parameters
    ----------
        J (np.ndarray): spin-spin iteration matrix, should be upper triangle
        h (np.ndarray, optional): local field

    """
    num_variables = J.shape[0]
    h = h if h is not None else np.zeros(num_variables)
    qubo = np.diag(2.0 * h)
    qubo += 4 * J
    qubo += np.diag(-2*J.sum(1))
    qubo += np.diag(-2*J.sum(0))
    return qubo

def qubo_ising(Q, a_ising=False, scale: float = 127.):
    """Change QUBO matrix to Ising matrix with linear terms as auxiliary spin \
    with offset == h.sum() / 2 + J.sum() / 4

    Parameters
    ----------
        Q (np.ndarray): QUBO matrix, should be upper triangle
        a_ising (bool, optional): if `True` then matrix with auxiliary spin \
            else return spin-spin iteration matrix and fields is returned
        scale (float, optional): Maximum matrix value

    Return:
    ----------
        (np.ndarray | tuple[np.ndarray,np.ndarray]]):
        Ising IM matrix - quadratic terms as an upper triangular matrix\
            with linear terms as auxiliary variable | \
            spin-spin iteraction matrix, magnetic field vector"""
    num_variables = Q.shape[0]
    J = np.zeros_like(Q)
    h = np.zeros(num_variables)
    off_diag_indices = np.triu_indices_from(Q, k=1)
    J[off_diag_indices] = 0.25 * Q[off_diag_indices]
    diag_mask = np.eye(num_variables, dtype=bool)
    h += Q[diag_mask] / 2
    h += np.triu(Q,k=1).sum(1)/4 + np.triu(Q,k=1).sum(0)/4
    if scale is not None:
        multipler = scale/max(np.abs(J).max(),np.abs(h).max())
        J *= multipler
        h *= multipler
    if a_ising:
        ising = np.zeros((Q.shape[0]+1, Q.shape[0]+1))
        ising[:-1, :-1] = J
        ising[:-1, -1] = h
        return ising
    return J, h

def load_graph(path,
               skiprows=1,
               ut=False):
    """Load graph adjesty matrix

    Parameters
    ----------
        path (str): path to file
        skiprows (int, optional):Defaults to 1.
        ut (bool, optional): Suggestion that matrix in file \
            is upper triangle (M_ij==M_ji). Defaults to False.

    Returns
    ----------
        np.ndarray: Matrix from graph
    """
    a = np.loadtxt(path, skiprows=skiprows)
    n = int(max(a[:,1].max(),a[:,0].max()))
    g_matrix = np.zeros((n, n))
    g_matrix[a[:,0].astype(int)-1, a[:,1].astype(int)-1] = a[:,2].astype(int)
    if ut:
        return g_matrix + np.triu(g_matrix, k=1).T
    return g_matrix
