
import libsbml
from astread import parse_expr
import sympy as sp
from sympy.utilities.lambdify import lambdastr
import scipy
import argparse

from astread import TIME_SYMBOL
from Compartment import *
from Specie import *
from Reaction import *
from Parameter import *
from Function import *

from types import FunctionType
from itertools import repeat

import jax
import jax.numpy as jnp
import sympy2jax

import math
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
#import optax
import jaxopt



class DiffEqSystem ():

    def __init__(self, sbml_mdl):
        self.mdl = sbml_mdl
        self.compartments = {} # map compartmentid -> Compartment
        self.reactions = {} # map reactionid -> Reaction
        self.species = {}   # map specieid -> Specie : for the variables species
        self.parametersspecies = {} #map specieid -> Specie : for the constant species that acts like parameters
        self.parameters = {} #map identifier -> Parameter (global parameters)
        self.functions = {} # map functionID -> Function (for the functions defined in the sbml)
        self.eqdiffsystem = {} #specieid -> expression: system of differential equations
        

        for compartment in sbml_mdl.getListOfCompartments():
            self.compartments[compartment.getId()] = Compartment(compartment)
        for function in sbml_mdl.getListOfFunctionDefinitions():
            self.functions[function.getId()] = Function(function)

        for param in sbml_mdl.getListOfParameters():
            self.parameters[param.getId()] = Parameter(param)

        for specie in sbml_mdl.getListOfSpecies():
            s = Specie(specie)
            if specie.getBoundaryCondition() or specie.getConstant():
                self.parametersspecies[specie.getId()] = s
            else:
                self.species[specie.getId()] = s


        for reaction in sbml_mdl.getListOfReactions():
            reactionid=reaction.getId()
            self.reactions[reactionid]=Reaction(reaction)
            for reactant in reaction.getListOfReactants():
                self.addReactionToSpecie(reactant.getSpecies(), reactionid)
            for product in reaction.getListOfProducts():
                self.addReactionToSpecie(product.getSpecies(), reactionid)
            for modifier in reaction.getListOfModifiers():
                self.addReactionToSpecie(modifier.getSpecies(), reactionid)
        self.build()
    
    def addReactionToSpecie (self, specieid, reactionid):
        try:
            self.species[specieid].addReaction(reactionid)
        except KeyError:
            if self.parametersspecies[specieid] != None:
                self.parametersspecies[specieid].addReaction(reactionid)
            else:
                raise KeyError(f"{specieid} in {reactionid} not registered in the list of species")

        
    def build (self):
        self.eqdiffsystem = {}

        newspecies = {}
        for specieid, specie in self.species.items():
            expr = specie.buildDifferentialEquation(self.reactions, self.species|self.parametersspecies, self.compartments)
            if expr == 0:
                self.parametersspecies[specieid] = specie
                print (f"{specieid} is constant and thus becomes a parameter.")
            elif expr.is_number:
                newspecies[specieid] = specie
                self.eqdiffsystem[specieid] = expr
            else:
                expr = self.expandFunctions(expr)
                if expr == 0:
                    self.parametersspecies[specieid] = specie
                    print (f"{specieid} is constant and thus becomes a parameter.")
                else:
                    newspecies[specieid] = specie
                    self.eqdiffsystem[specieid] = expr
        self.species = newspecies

        for specieid, expr in self.eqdiffsystem.items():
            print (specieid,": ", expr)
        
        print ("rewriting:")
        self.rewriteSymbols()

        odes = sympy2jax.SymbolicModule(list(self.eqdiffsystem.values()))
        params = [str(param.symbol)[1:] for param in self.listParameters()]
        species = [str(specie.symbol)[1:] for specie in self.species.values()]

        self.odes = ODETerm(lambda t,y,args : jnp.stack(odes(**dict(zip(species, y)), **args)))

        print (f"Found {len(self.species)} specie(s), {len(self.reactions)} reaction(s), {len(self.parameters)} parameter(s), {len(self.parametersspecies)} specie(s) considered as parameter(s) and {len(self.functions)} function definition(s).")

       
    def rewriteSymbols(self):
        """
        Rewrites the ODEs with unique symbols, in order to avoid shadowing declarations.
        Also converts the symbols (that are all constant functions) into time-dependent functions (for
        species) and constant symbols (for parameters)
        """
        compartmentsDict = {sp.Function(sym.getId())():sym.symbol for sym in self.compartments.values()}
        speciesDict = {sp.Function(sym.getId())():sym.symbol for sym in self.species.values()}
        paramspeciesDict = {sp.Function(sym.getId())():sym.symbol for sym in self.parametersspecies.values()}
        parameters = {sp.Function(sym.getId())():sym.symbol for sym in self.parameters.values()}
        for expr in self.eqdiffsystem:
            self.eqdiffsystem[expr] = (self.eqdiffsystem[expr].subs(compartmentsDict | speciesDict | paramspeciesDict |parameters))
            print ("\t",self.eqdiffsystem[expr])

        

    def expandFunctions(self, expression):
        while True: #because there is no infinite recursions in python (nor do-until syntax) -_-
            recursive_call = False
            for func in expression.atoms(sp.Function):
                name = func.func.__name__
                if name in self.functions:
                    expression = self.functions[name].apply(expression)
                    recursive_call = True
            if recursive_call == False:
                break
        return expression

    def writeDiffEqs (self, filename):
        symbolmap = {}
        count = 1
        for param in self.listParameters():
            symbolmap [param.symbol] = sp.symbols("k"+str(count))
            count += 1
        count =1
        speciemap = {}
        for specie in self.species.values():
            symbolmap[specie.symbol] = sp.symbols("x"+str(count))
            speciemap[specie.getId()] = sp.symbols("x"+str(count))
            count += 1

        neqs = {}
        with open(filename, "w") as file:
            for var, eq in self.eqdiffsystem.items():
                varsym = speciemap[var]
                neq = eq.subs(symbolmap)
                file.write(f"{varsym}: {neq}\n")
        

    def getY0(self, usr_paramsvals={}, usr_speciesvals={}):
        """
            Generates a function f :: t -> x -> f(t,x)=dx/dt as well as the initial point
            The user can modify the values of some parameters and the initial value of some species
            returns the value of y0 and the values of the parameters
        """
        params = self.listParameters()
        paramsvals = {}
        speciesvals = []
        x = [specie.symbol for specie in self.species.values()]
        speciessymb = {sp.Function(specie.symbol)(TIME_SYMBOL):specie.symbol for specie in self.species.values()}

        for param in params:
            name = str(param.symbol)[1:]
            if param.getId() in usr_paramsvals:
                paramsvals[name] = jnp.array(usr_paramsvals[param.getId()])
            else:
                paramsvals[name] = jnp.array(param.initialvalue)

        for specie in self.species.values():
            if specie.getId() in usr_speciesvals:
                speciesvals.append(usr_speciesvals[specie.getId()])
            else:
                speciesvals.append(specie.initialvalue)

        return jnp.array(speciesvals), paramsvals

         
        
    def getTrajectory (self, y0, params, ts, t0=0,dt0=0.1):

        saveat = SaveAt (ts=ts)
        solver = Dopri5()
        t1 = ts[-1]

        sol = diffeqsolve(self.odes, solver, t0=t0, t1=t1, dt0=dt0, saveat=saveat, y0=y0, args=params)
        return sol.ys


    def adjust (self, y0, params0, ts, targets, t0=0, dt0=0.1, lr=0.2, iters=300):
        #optimizer = optax.adabelief(lr)
        #opt_state = optimizer.init(params0)

        times = jnp.array(ts)

        def weight_fun(t, alpha=10):
            return jnp.exp(-alpha*t**2)
        @jax.jit
        def loss_fn (params, alpha=1):
            weights = weight_fun(times,alpha)[:,None]
            pred = self.getTrajectory(y0, params, ts, t0=t0, dt0=dt0)
            return jnp.mean(weights*(pred - targets))
            #return jnp.mean((pred - targets)**2)
        solver = jaxopt.BFGS(loss_fn, implicit_diff=True, maxiter=iters)
        ret = solver.run(params0)
        return ret.params
        #@jax.jit
        #def step (params, opt_state):
        #    loss, grads = jax.value_and_grad(loss_fn)(params)
        #    updates, opt_state = optimizer.update(grads, opt_state)
        #    params = optax.apply_updates(params, updates)
        #    return params, opt_state, loss

        #params = params0
        #for i in range(iters):
        #    params, opt_state, loss = step(params, opt_state)
        #    if i % 10 == 0:
        #        print (f"step={i}: {loss}")
        #return params
        




            

    def listParameters (self):
        params = []

        for param in self.parameters.values():
           params.append(param)
       
        for compartment in self.compartments.values():
            params.append(compartment)

        for reaction in self.reactions.values():
            for param in reaction.localparameters.values():
                params.append(param)
       
        for param in self.parametersspecies.values():
            params.append(param)

        return params






#mdl = libsbml.readSBML("examples/607.xml").getModel()

#mdl = libsbml.readSBML("examples/BIOMD0000000056_decomposition_r1.xml").getModel()
#mdl = libsbml.readSBML("examples/BIOMD0000000095_decomposition_r1.xml").getModel()
#mdl = libsbml.readSBML("examples/BIOMD0000000096_decomposition_r1.xml").getModel()
#mdl = libsbml.readSBML("example1.xml").getModel()
