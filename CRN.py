

import libsbml

import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=30' # Use multiples CPU devices
#os.environ["XLA_FLAGS"]="--xla_cpu_multi_thread_eigen=trueintra_op_parallelism_threads=1 --xla_force_host_platform_device_count=8"

import jax
jax.config.update("jax_enable_x64", True)
from functools import partial
from enum import Enum
from astread import TIME_SYMBOL
from Compartment import *
from Specie import *
from Reaction import *
from Parameter import *
from Rule import *

from Function import *


import networkx as nx
import jax.numpy as jnp
from astread import parse_expr

from diffrax import diffeqsolve, ODETerm, Kvaerno3, Kvaerno4, Kvaerno5, SaveAt, PIDController, ImplicitEuler
import diffrax
import optimistix as optx
import lineax
import scipy
import scipy.integrate as si

import sys
jnp.set_printoptions(threshold=sys.maxsize)



def cast_to_float64(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), tree)
def cast_module_to_float64(module):
    for attr in dir(module):
        value = getattr(module, attr)
        if isinstance(value, jnp.ndarray):
            setattr(module, attr, jnp.asarray(value, dtype=jnp.float64))
    return module

@partial(jax.jit, static_argnums=[0,4,5])
def computeTrajectoryWithSolverTs(odes, y0,p0, ts, solver, max_steps=4096):
    t1=ts[-1]
    function = ODETerm(lambda t,y,args: odes(y, args))
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-9, pcoeff=0.1, icoeff=0.2, dcoeff=0)
    saveat = SaveAt(t0=True, t1=True, ts=jnp.array(ts))
    sol = diffeqsolve(function, solver, t0=0, t1=t1, dt0=None, #dt0=0.1, 
                      y0=y0, 
                      saveat=saveat,
                      stepsize_controller=stepsize_controller,
                      args=p0,
                      throw=False,
                      adjoint=diffrax.ForwardMode(),
                      max_steps=max_steps)

    return sol.ys
@partial(jax.jit, static_argnums=[0,4,5])
def computeTrajectoryWithSolverSteps(odes, y0,p0, ts, solver, max_steps=4096):
    t1=ts
    function = ODETerm(lambda t,y,args: odes(y, args))
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-9, pcoeff=0.1, icoeff=0.2, dcoeff=0)
    saveat = SaveAt(t0=True, t1=True, steps=True)
    sol = diffeqsolve(function, solver, t0=0, t1=ts, dt0=None,
                      y0=y0, 
                      saveat=saveat,
                      stepsize_controller=stepsize_controller,
                      args=p0,
                      throw=False,
                      adjoint=diffrax.ForwardMode(),
                      max_steps=max_steps)

    return sol
def computeTrajectoryWithSolver(odes,y0,p0,ts,solver,max_steps=4096):
    if isinstance(ts, list):
        return computeTrajectoryWithSolverTs(odes,y0,p0,ts,solver,max_steps)
    else:
        nfeats = len(y0)
        sol = computeTrajectoryWithSolverSteps(odes,y0,p0,ts,solver,max_steps)

        times = sol.ts[jnp.isfinite(sol.ts)]
        
        ret = sol.ys[jnp.isfinite(sol.ys)]
        ret = ret.reshape(-1,nfeats) 
        return times, ret




def lsoda (odes, y0, p0, ts):
    function = lambda t,y: odes(y,p0) 
    ret = si.solve_ivp(function,(0,ts[-1]), y0, method="LSODA", t_eval=ts)
    return ret




def showParam (params, pdict):
    ndict = {pdict[p[0]]:p[1] for p in params.items()}
    return ndict

class VertexType (Enum):
    SPECIE=0
    REACTION=1

class CRNGraph:

    def __init__(self, sbml_mdl):


        self.sr = nx.DiGraph() #specie-reaction graph

        self.compartments = {} # map compartmentid -> Compartment
        self.reactions = {} # map reactionid -> Reaction
        self.species = {}   # map specieid -> Specie : for the variables species
        self.parametersspecies = {} #map specieid -> Specie : for the constant species that acts like parameters
        self.parameters = {} #map identifier -> Parameter (global parameters)
        self.functions = {} # map functionID -> Function (for the functions defined in the sbml)
        self.rules = {"assignment":{}, "rate":{}, "algebraic":{}}

        self.initialassignments = {}
        for initialassignment in sbml_mdl.getListOfInitialAssignments():
            self.initialassignments[initialassignment.getSymbol()] = initialassignment
 
        for compartment in sbml_mdl.getListOfCompartments():
            self.compartments[compartment.getId()] = Compartment(compartment)

        for function in sbml_mdl.getListOfFunctionDefinitions():
            self.functions[function.getId()] = Function(function)

        for param in sbml_mdl.getListOfParameters():
            self.parameters[param.getId()] = Parameter(param)

        for specie in sbml_mdl.getListOfSpecies():
            s = Specie(specie)
            specieid = s.getId()
            self.sr.add_node(specieid, bipartite=VertexType.SPECIE)
            if specie.getBoundaryCondition() or specie.getConstant():
                self.parametersspecies[specieid] = s
            else:
                self.species[specieid] = s

        for rule in sbml_mdl.getListOfRules():
            if rule.isAssignment():
                self.rules["assignment"][rule.getVariable()] = Rule(rule)
            elif rule.isAlgebraic():
                self.rules["algebraic"][rule.getIdAttribute()]=Rule(rule)
                print ("[Warning] algebraic rule found.")
            elif rule.isRate():
                self.rules["rate"][rule.getIdAttribute()]=Rule(rule)
            else:
                raise ValueError ("Unimplemented rule managment")
        print ("Rules found ",len(self.rules["assignment"]), " assigment, ", len(self.rules["algebraic"])," algebraic, ", len(self.rules["rate"]), " rate.")
        #symbol map to rewrite expressions using unique symbols
        pdict = {sp.Function(sym.getId())():sym.symbol for sym in self.parameters.values()}
        cdict = {sp.Function(sym.getId())():sym.symbol for sym in self.compartments.values()}
        sdict = {sp.Function(sym.getId())():sym.symbol for sym in self.species.values()}
        spdict = {sp.Function(sym.getId())():sym.symbol for sym in self.parametersspecies.values()}
        ruledict = {sp.Function(name)():rule.formula for name, rule in self.rules["assignment"].items()}
        symdict = pdict | cdict | sdict | spdict

        self.symdict = symdict



        for reaction in sbml_mdl.getListOfReactions():
            reactionid=reaction.getId()
            self.sr.add_node(reactionid, bipartite=VertexType.REACTION)
            self.reactions[reactionid]=Reaction(reaction, self.functions, symdict, self.rules["assignment"])

            for reactant in reaction.getListOfReactants():
                sid = reactant.getSpecies()
                self.sr.add_edge(sid, reactionid, type="reactant", stoichiometry=reactant.getStoichiometry())
            for product in reaction.getListOfProducts():
                sid = product.getSpecies()
                self.sr.add_edge(reactionid, sid, type="product", stoichiometry=product.getStoichiometry())
            for modifier in reaction.getListOfModifiers():
                sid = modifier.getSpecies()
                self.sr.add_edge(sid,reactionid, type="modifier")

        self.identifyConstantSpecies()
        print (f"Found {len(self.species)} species, {len(self.parameters)} global parameters, {len(self.compartments)} compartments, {len(self.reactions)} reactions and {len(self.parametersspecies)} constant species.")

        
    def identifyConstantSpecies(self):
        species = {}
        for specieid, specie in self.species.items():
            expr = specie.buildDifferentialEquation(self.sr, 
                                                    self.species|self.parametersspecies,
                                                    self.reactions, 
                                                    self.compartments)
            if expr == 0:
                self.parametersspecies[specieid] = specie
            else:
                species[specieid] = specie

        self.species = species


            
    def listParameters(self):
        params = []
        params += list(self.parameters.values())
        for r in self.reactions.values():
            params+= list(r.localparameters.values())
        params += list (self.compartments.values())
        params += list (self.parametersspecies.values())
        return params

    def rewrite(self):
        odes = [specie.buildDifferentialEquation(self.sr,
                                                 self.species | self.parametersspecies,
                                                 self.reactions,
                                                 self.compartments,
                                                 symbolic=True) for specieid, specie in self.species.items()]
        params = self.listParameters()
        symbs = {}
        i = 1
        for p in params:
            symbs[p.symbol] = sp.symbols("k"+str(i))
            i = i+1
        i=1
        for s in self.species.values():
            symbs[s.symbol] = sp.symbols("x"+str(i))
            i = i+1

        exprs = [ode.subs(symbs) for ode in odes]
        for expr in exprs:
            print (expr)

    def generateODEs(self):
        exprs = []
        for specie in self.species.values():
            expr = specie.buildDifferentialEquation(self.sr,
                                                    self.species|self.parametersspecies,
                                                    self.reactions,
                                                    self.compartments)
            exprs.append(expr)
        entities = {entitie.getId():entitie for entitie in list(self.species.values())+self.listParameters()}
        self.initializesEntities(entities.values())
        ssymb,sval = zip (*[(str(specie.symbol)[1:], specie.val) for specie in self.species.values()])
        ssymb, sval = list(ssymb), list(sval)
        variables = {}
        for rule in self.rules["rate"].values():
            print ("Found rateOf rule concerning ", rule.getVariable())
            var = entities[rule.getVariable()]
            variables[var.symbol] = var.getId()

            expr = parse_expr(rule.getMath())
            expr = self._expandFunctions(self.functions, expr)
            expr = self._expandFunctions(self.rules["assignment"],expr)
            expr = expr.subs(self.symdict)
            exprs.append(expr)

        snames = [specie.getId() for specie in self.species.values()]
        psymb, pval, pnames = [], [], []
        vsymb, vval, vnames = [], [], []
        for param in self.listParameters():
            if not param.symbol in variables:
                psymb.append(str(param.symbol)[1:])
                pval.append (param.val)
                pnames.append(param.getId())
            else:
                vsymb.append(str(param.symbol)[1:])
                vval.append(param.val)
                vnames.append(param.getId())
                

        odes = cast_module_to_float64(sympy2jax.SymbolicModule(exprs))
        dict_function = lambda y, params: dict(zip(ssymb+vsymb, y)) | params
        self.odes = lambda y, params: cast_to_float64(jnp.stack(odes(**dict_function(y,params))))
        return self.odes, jnp.array(sval+vval, dtype=jnp.float64), dict(zip (psymb, jnp.array(pval, dtype=jnp.float64))), dict(zip(psymb, pnames))
    
    @partial(jax.jit, static_argnames=["self","ode_solver", "n_step"])
    def adjust(self, y0, p0, ts, target, n_step=10, ode_solver=Kvaerno5(), ode_max_steps=4096):
          
        loss_fn = lambda params,_: jnp.mean((computeTrajectoryWithSolver(self.odes, y0, params, ts, ode_solver, max_steps=ode_max_steps) - target)**2)
        solver = optx.LevenbergMarquardt(
                    rtol=1e-8, atol=1e-8,
                    #verbose=frozenset({"step", "accepted","loss", "step_size"})
                )

        res = optx.least_squares(loss_fn, solver, p0, max_steps=n_step, options={"autodiff_mode":"fwd"}, throw=False)
        return res.value


    def simulate(self,ts, solver=Kvaerno5(), max_steps=4096, return_times = False):
        f, y0, p0, names = self.generateODEs()
        times, traj = computeTrajectoryWithSolver(f, y0, p0,ts,solver, max_steps=max_steps)

        if return_times:
            return times, traj
        else:
            return traj

    def initializesEntities (self, entities):
        """
            Identifies entities for which the initial value are functions of the initial values of other entities
        """
        initialized, tocompute = {}, {}
        g = nx.DiGraph()

        for e in entities:
            if e.getId() in self.initialassignments:
                expr = parse_expr(self.initialassignments[e.getId()].getMath())
                expr = self._expandFunctions(self.functions, expr)
                expr = self._expandFunctions(self.rules["assignment"], expr)
                tocompute[e.symbol] = (e,expr.subs(self.symdict))
                g.add_node(e.symbol, mark=False)
            else:
                initialized[e.symbol] = e.val
                g.add_node(e.symbol, mark=True)

        for symb, (_,expr) in tocompute.items():
            for atom in expr.atoms(sp.Symbol):
                g.add_edge(atom, symb)

        if tocompute != {}:
            print ("Applying initialAssignments:")
        while True:
            finished=True
            newtocompute = {}
            for symb, (e,expr) in tocompute.items():
                ancestors = g.predecessors(symb)
                if all ([g.nodes[node]["mark"] for node in ancestors]):
                    initialized[symb] = expr.subs(initialized)
                    e.val = initialized[symb]
                    print ("assigning: ",symb, "=", initialized[symb])
                    g.nodes[symb]["mark"]=True
                    finished = False
                else:
                    newtocompute[symb]=expr
            tocompute = newtocompute
            if finished == True:
                break
        if tocompute != {}:
            raise ValueError ("Unable to compute initialAssignments for ", tocompute)


    def _expandFunctions(self, macros, expression):
        """
            expands the macros in the current expression.
            macros should be a map symbol -> object implementing the method apply [TODO should be a callable]
        """
        while True: #because there is no infinite recursions in python (nor do-until syntax) -_-
            recursive_call = False
            for func in expression.atoms(sp.Function):
                name = func.func.__name__
                if name in macros:
                    expression = macros[name].apply(expression)
                    recursive_call = True
            if recursive_call == False:
                break
        return expression


def test(solver=Kvaerno5(root_finder=optx.Chord(rtol=1e-8,
                                                atol=1e-8,
                                                linear_solver=lineax.AutoLinearSolver(well_posed=False))),
         ts=[1,2,3]):
    x =  CRNGraph(libsbml.readSBML("biomodels/BIOMD0000000005.xml").getModel())
    f, y0, p0, names = x.generateODEs()
    traj = computeTrajectoryWithSolver(x.odes, y0, p0, ts, solver)
    pinit = {p[0]:jnp.array(0.01, dtype=jnp.float64) for p in p0.items()}
    #pinit = {p[0]:p[1]+100 for p in p0.items()}
    return x,y0,p0,pinit,traj,solver,names


def generate_data(filename, t1):
    x =  CRNGraph(libsbml.readSBML(filename).getModel())
    return x.simulate(t1)


import time

def timeit (f):
    start = time.time()
    f()
    end = time.time()
    print ("elapsed: ", end-start)


