This repository contains an implementation of an Ising machine based on a optoelectronic oscillator with feedback for solving combinatorial optimization problems

```
git clone https://github.com/Schmidtid/ising_machine.git
cd ising_machine
pip install -r requriments.txt
```

Options are offered for comparison based on periodic function, polynomial, sigmoid, piecewise constant (clip). Some modifications are also proposed for IM based on an optoelectronic oscillator (periodic) for QUBO tasks.

To find some minimum of qubo matrix you can try:

```
import ising_machine as im
g = im.load_graph('ising_machine/data/be150.3.8.sparse', ut=True)
simulator = Solver(g)
x = simulator.mzm_ising() # Ising machine based on Mach-Zender modulator
simulator._obj() # print QUBO and Ising energy
x_0 = simulator.clip_fb(min_eigv=13, h_max=1) for problem with big coefficient
simulator._obj()
x_1 = simulator.simulated_annealing() # some heuristic algorithm
simulator._obj()
```

Problems can also be imported from a `.lp` or `.mps` file and solved using CPLEX/SCIP.

```
import ising_machine as im
problem = im.convert('ising_machine/data/QPLIB_3750.lp')
Q = problem.to_qubo(eq=1300) # if ineq_type=True, unbalanced penalization is used for inequlity constraints
simulator = im.Solver(Q)
x1 = simulator.simulated_annealing() 
x2 = simulator.clip_fb(min_eigv=13,h_max=5,n_runs=10) # you can set device='cuda'
for i in [x1, x2]:
    problem.qubo.check(i)
problem.solve('SCIP') # or 'CPLEX'
```
You can read more about unbalanced penalization [here](https://arxiv.org/pdf/2211.13914)

You can also express the formulation in the terms of the Ising Hamiltonian

```
>>> import ising_machine as im
>>> J = im.load_graph('ising_machine/data/G18.txt')
>>> simulator = im.Solver(J, h=None, values='spin')
>>> x = simulator.mzm_ising()
>>> print(x)
[1. 1. -1. ... 1. -1. 1.]
```
