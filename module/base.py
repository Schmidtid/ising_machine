import numpy as np
from docplex.mp.linear import LinearExpr


class Problem(object):
    """Problem holder and transformer to QUBO"""
    _constraint_list: list[LinearExpr] = []
    def __init__(self,
                 linear:np.ndarray = None,
                 quadratic:np.ndarray = None,
                 equality_matrix:np.ndarray = None,
                 equality_vector:np.ndarray = None,
                 inequality_matrix:np.ndarray = None,
                 inquality_vector:np.ndarray = None,
                 constant: float = 0.0,
                 sense: float = 1.) -> None:
        """
        Parameters
        ----------
            linear (np.ndarray, optional): Linear terms of objective function.
            quadratic (np.ndarray, optional): Quadratic terms of objective function.
            equality_matrix (np.ndarray, optional): Matrix variables coefficient in constraints Ax==b.
            equality_vector (np.ndarray, optional): vector of equality constraints.
            inequality_matrix (np.ndarray, optional): less-or-equal inequality constraints matrix.
            inquality_vector (np.ndarray, optional): inequality constraints vector.
            constant (float, optional): Constant of objective function.
            sense (float, optional): 1 to Minimaze or -1 to Maximize
        """
        self.A = equality_matrix
        self.G = inequality_matrix
        self.b = equality_vector
        self.h = inquality_vector
        self.quadratic = quadratic
        self.linear = linear
        self.c = constant
        self.sign = sense

    def __repr__(self) -> str:
        res = f"Problem with {max((self.linear.shape[0], self.quadratic.shape[0]), default=0)} var."
        if self.A.size>0:
            res += f'\n{self.b.size} Equality'
        if self.G.size>0:
            res += f' and {self.h.size} Inequality'
        res += ' constraints'
        return res

    def _nvar(self):
        """Get number of binary variables"""
        if self.G.size:
            return self.G.shape[1]
        return self.linear.size or self.quadratic.shape[0]

    def _ineq_to_eq(self):
        slack = np.multiply(self.G,self.G>0).sum(1)-self.h
        sign = -np.sign(slack)
        n_slack = np.ceil(np.log2(np.abs(slack + 1)))
        new_G = np.zeros((self.G.shape[0],int(self.G.shape[1]+n_slack.sum())))
        new_G[:,:self.G.shape[1]] = self.G
        n_slack = n_slack.astype(int)
        ind = self.G.shape[1]
        for i in range(new_G.shape[0]):
            _ = []
            for j in range(n_slack[i]):
                _.append(sign[i] * 2**j)
            _[-1] += sign[i] * (slack[i]-2**(n_slack[i]-1))
            for x, num in enumerate(range(ind,ind+n_slack[i])):
                new_G[i,num] = _[x]
            ind += n_slack[i]
        self.G = new_G

    def _unb_ineq(self, strength_ineq: list):
        strength_ineq = strength_ineq or [0.95, 0.0351]
        n = self._nvar()
        penalty = np.zeros((n, n), dtype=np.float64)
        for const in self._constraint_list:
            new_exp = const.get_right_expr() + -1 * const.get_left_expr()
            expr = -strength_ineq[0] * new_exp + strength_ineq[1] * new_exp**2
            for x, weight in expr.get_linear_part().iter_terms():
                penalty[x.index, x.index] += weight
            for x, y, weight in expr.iter_quad_triplets():
                penalty[x.index, y.index] += weight
        return penalty

    def to_qubo(self, eq: float = 1.2,
                n_eq: float = 1.2,
                ineq_type: bool = False,
                scale: float = None,
                strength_ineq: list = None):
        """Convert linear constraints to qubo 

        Args:
            eq (float, optional): Penalty terms for equality constraints.
            n_eq (float, optional): Penalty terms for equality constraints.
            ineq_type (bool, optional): If `True`, then the unbalanced method is used. \
                In another case, slack variables are used.
            scale (float, optional): Multiplier for objective function.
            strength_ineq (list[float], optional): Unbalanced penalty.


        Returns:
            np.ndarray: Qubo matrix
        """
        n = self._nvar()
        qubo = np.zeros((n, n),dtype=np.float64)
        qubo += eq * (self.A.T@self.A - 2*np.diag(self.b.T@self.A))
        if ineq_type:
            self._ineq_to_eq()
            n_new = self._nvar()
            qubo1 = np.zeros((n_new, n_new), dtype=np.float64)
            qubo1 += n_eq * (self.G.T@self.G - 2*np.diag(self.h.T@self.G))
            qubo1[:n, :n] += qubo
            qubo = qubo1.copy()
        else:
            qubo += self._unb_ineq(strength_ineq or [0.95, 0.0351])
        scale = scale if scale else 1/max((self.linear.max(), self.quadratic.max()))
        qubo[:n, :n] += scale * (np.diag(self.linear) + self.quadratic)
        return qubo

    def check(self,x, tol=1e-8):
        """Check constraints

        Parameters
        ----------
            x (array): Decision binary vector
            tol (float, optional): Tollerance. Defaults to 1e-8.

        Returns:
        ----------
            None|float: if solution is correct, then objectiv value is returned
        """
        res = {'linear_ineq':0,
               'linear_eq':0}
        if self.G.size>0:
            res['linear_ineq'] = (~(self.G.dot(x).T<=self.h)).sum()
        if self.A.size>0:
            res['linear_eq'] = (np.abs(self.A.dot(x[:self.A.shape[1]]) - self.b)>tol).sum()
        print(f"Linear:\n\tEq.: {res['linear_eq']}\n\t'Ineq.: " +
              f"{res['linear_ineq']}\nQuadratic:\n\tEq.: ")
            #   f"{res['quad_eq']}\n\tIneq.: {res['quad_ineq']}")
        if sum(res.values())==0:
            return x.T@self.quadratic@x+self.linear@x.T+self.c
        return None
