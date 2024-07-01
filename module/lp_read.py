"""Module for work with .lp//mps files"""
import warnings
import numpy as np
from pyscipopt import Model
from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr
from docplex.mp.model_reader import ModelReader
from ising_machine.module.base import Problem


class convert:
    """Convert .lp or .mps files to QUBO Problem"""
    def __init__(self, filename: str) -> None:
        self._model = ModelReader.read_model(filename)
        self.problem = Model(self._model.name)
        self.problem.readProblem(filename)
        self._var_names = {}
        self._var_bounds = {}
        self._quadratic_const = []
        self.qubo: Problem = None

    @staticmethod
    def _linear_expr(expr):
        linear = {}
        for x, coeff in expr.iter_terms():
            linear[x.index] = coeff
        return linear

    @staticmethod
    def _transfer(left_dict, right_dict):
        result_dict = {}
        for key in left_dict.keys():
            if key in right_dict:
                result_dict[key] = left_dict[key] - right_dict[key]
            else:
                result_dict[key] = left_dict[key]

        for key in right_dict.keys():
            if key not in left_dict:
                result_dict[key] = -right_dict[key]
        return result_dict

    @staticmethod
    def _quadratic_expr(expr):
        linear = convert._linear_expr(expr.get_linear_part())
        quad = {}
        for x, y, coeff in expr.iter_quad_triplets():
            i = x.index
            j = y.index
            quad[i, j] = coeff
        return linear, quad

    @staticmethod
    def _trival_const(expr:LinearExpr):
        """Check trivial constraints like `x+y << z`

        Parameters
        ----------
            expr (docplec.mp.linear.LinearExpr)

        Returns:
        ----------
            Literal[0, 1]
        """
        lh = 0.
        uh = 0.
        for x, coef in expr.left_expr.iter_sorted_terms():
            uh += max(0, coef) # * problem._var_bounds[x.index][1]
            lh += min(0, coef) # * problem._var_bounds[x.index][0]
        for x, coef in expr.right_expr.iter_sorted_terms():
            uh += max(0, -coef) # * problem._var_bounds[x.index][1]
            lh += min(0, -coef) # * problem._var_bounds[x.index][0]
        rhs = expr.cplex_num_rhs()
        if expr.sense.value == 1:
            if uh < rhs:
                return 0
        elif expr.sense.value == 3:
            if lh > rhs:
                return 0
        return 1

    def _lc(self, constraint):
        left_expr = constraint.get_left_expr()
        right_expr = constraint.get_right_expr()
        if isinstance(left_expr, Var):
            left_expr = left_expr + 0
        left_linear = self._linear_expr(left_expr)
        if isinstance(right_expr, Var):
            right_expr = right_expr + 0
        if constraint.sense.value == 3:
            left_expr *= -1
            right_expr *= -1
            constraint.set_sense('LE')
        right_linear = self._linear_expr(right_expr)
        linear = self._transfer(left_linear, right_linear)
        rhs = constraint.cplex_num_rhs()
        return (linear, constraint.sense.value-1, rhs)

    def _qc(self, constraint):
        left_expr = constraint.get_left_expr()
        right_expr = constraint.get_right_expr()
        if isinstance(left_expr, Var):
            left_expr = left_expr + 0
        if isinstance(right_expr, Var):
            right_expr = right_expr + 0
        if constraint.sense.value == 3:
            left_expr *= -1
            right_expr *= -1
            constraint.set_sense('LE')
        if left_expr.is_quad_expr():
            left_lin, left_quad = self._quadratic_expr(left_expr)
        else:
            left_lin = self._linear_expr(left_expr)
            left_quad = {}
        if right_expr.is_quad_expr():
            right_lin, right_quad = self._quadratic_expr(right_expr)
        else:
            right_lin = self._linear_expr(right_expr)
            right_quad = {}
        linear = self._transfer(left_lin, right_lin)
        quadratic = self._transfer(left_quad, right_quad)
        rhs = constraint.cplex_num_rhs()
        self._quadratic_const.append((linear, quadratic, constraint.sense.value-1, rhs))

    def get_obj(self):
        """Get objective function from docplex.mp.Model
        
        Returns:
        ----------
            tuple[Literal[-1, 1], np.ndarray, np.ndarray, float]: Sense, \
                linear terms, quadratic terms, objective constant"""
        cvar = False
        for x in self._model.iter_variables():
            self._var_names[x] = x.name
            self._var_bounds[x.index] = (x.lb, x.ub)
            if x.is_continuous():
                cvar = True
        if cvar:
            warnings.warn("Currently IM working only with discrete variablies",stacklevel=3)
        sense = 1.
        if self._model.objective_sense.value == 2:
            sense = -1.
        quadratic = {}
        if self._model.objective_expr.is_quad_expr():
            linear, quadratic = self._quadratic_expr(self._model.objective_expr)
        else:
            linear = self._linear_expr(self._model.objective_expr.linear_part)
        lin = np.zeros(len(self._var_names), dtype=np.float64)
        quad = np.zeros((len(self._var_names), len(self._var_names)), dtype=np.float64)
        for x, coeff in linear.items():
            lin[x] = coeff
        for x, coeff in quadratic.items():
            i, y = x
            quad[i, y] = coeff
        return sense, lin, quad, self._model.objective_expr.constant

    def const(self):
        """Get constraints from docplex.mp.Model

        Returns:
        ----------
            tuple[np.ndarray, np.ndarray, np.ndarray]: Constraints matrix, \
                constraints vector, constraint type
        """
        linear_const = []
        for const in self._model.iter_linear_constraints():
            if self._trival_const(const):
                linear_const.append(self._lc(const))
        if self._model.number_of_quadratic_constraints:
            warnings.warn("Only linear constraints currently avaliable",stacklevel=2)
        for const in self._model.iter_quadratic_constraints():
            self._qc(const)

        left_side = np.zeros((len(linear_const), len(self._var_names)), dtype=np.float64)
        right_side = np.zeros(len(linear_const), dtype=np.float64)
        senses = np.zeros(len(linear_const),dtype=bool)
        for i, triplet in enumerate(linear_const):
            d, condition, rhs = triplet
            senses[i] = bool(condition)
            right_side[i] = rhs
            for key, value in d.items():
                left_side[i, key] = value
        return left_side, right_side, senses

    def to_qubo(self, *args, **kward):
        """Convert linear constraints to qubo 

        Parameters
        ----------
            eq (float, optional): Penalty terms for equality constraints.
            n_eq (float, optional): Penalty terms for equality constraints.
            ineq_type (bool, optional): If `True`, then the unbalanced method is used. \
                In another case, slack variables are used.
            scale (float, optional): Multiplier for objective function.
            strength_ineq (list[float], optional): Unbalanced penalty.

        Returns:
        ----------
            np.ndarray: Qubo matrix
        """
        sense, lin, quad, const = self.get_obj()
        left_c, right_c, senses = self.const()
        model = Problem(lin, quad,
                       left_c[senses], right_c[senses],
                       left_c[~senses], right_c[~senses],
                       const, sense)
        self.qubo = model
        self.qubo._constraint_list = [i
                                      for i in self._model.iter_linear_constraints()
                                      if i.sense.value==1]
        return model.to_qubo(*args, **kward)

    def solve(self,solver:str='SCIP'):
        """
        Parameters:
        ----------
            solver (str, optional): Solver type. Currently in ['SCIP', 'CPLEX']
        
        Raises: CPLEX Error

        Returns:
            solution | solution dict
        """
        if solver == 'CPLEX':
            if not (self.problem.getNVars() > 1000 or
                self.problem.getNConss() > 1000):
                try:
                    self._model.solve()
                    return self._model.solution
                except Exception as e:
                    raise e
        if solver == 'SCIP':
            self.problem.optimize()
            return self.problem.getVarDict()

    def _check(self, x):
        if len(x) != len(self._var_names):
            raise IndexError(
                f"The size of `x`: {len(x)}, does not match the number of problem variables: "
                f"{len(self._var_names)}")

        # violated_variables = []
        # for i, val in enumerate(x):
        #     variable = self.get_variable(i)
        #     if val < variable.lowerbound or variable.upperbound < val:
        #         violated_variables.append(variable)

        # violated_constraints = []
        # for constraint in cast(List[Constraint], self._linear_constraints) + cast(
        #     List[Constraint], self._quadratic_constraints
        # ):
        #     lhs = constraint.evaluate(x)
        #     if constraint.sense == ConstraintSense.LE and lhs > constraint.rhs:
        #         violated_constraints.append(constraint)
        #     elif constraint.sense == ConstraintSense.GE and lhs < constraint.rhs:
        #         violated_constraints.append(constraint)
        #     elif constraint.sense == ConstraintSense.EQ and not isclose(lhs, constraint.rhs):
        #         violated_constraints.append(constraint)

        # feasible = not violated_variables and not violated_constraints
