
from DiffExpr import *
import libsbml
import numpy as np
import jax.numpy as jnp
import torch

def getSystem(filename):
    file = libsbml.readSBML(filename)
    if file == None:
        print ("File not found")
        exit(0)
    mdl = file.getModel()    
    return DiffEqSystem(mdl)
def run(filename, targetname, ts, t0=0, dt0=0.1):
    system = getSystem(filename)
    y0, params = system.getY0()
    traj = system.getTrajectory(y0,params,ts, t0=t0, dt0=dt0)
    print (traj)
    traj = torch.tensor(np.array(traj))
    torch.save(traj, targetname)
    return traj


def getInfos(system):
    y0, params = system.getY0()
    if args.parameters_values == True:
        with np.printoptions(precision=3,suppress=True):
            #vals = list(np.array(jnp.stack(list(params.values()))))
            vals = {param.getId():float(params[str(param.symbol)[1:]]) for param in system.listParameters()}
            print ("Parameter values are: ", vals)

def optimize(system, ts, t0, dt0, lr, iters):
    y0, params = system.getY0()
    targets = system.getTrajectory(y0, params, ts)
    params0 = {param:val - 1e-1 for param, val in params.items()}
    params = system.adjust(y0, params0, ts, targets, t0=t0, dt0=dt0, lr=lr, iters=iters)
    traj = system.getTrajectory(y0, params, ts, t0=t0, dt0=dt0)
    print ("trajectory:")
    print (traj)
    with np.printoptions(precision=3,suppress=True):
        #vals = list(np.array(jnp.stack(list(params.values()))))
        vals = {param.getId():float(params[str(param.symbol)[1:]]) for param in system.listParameters()}
        print ("Parameter values are: ", vals)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("-i", "--input-file", help="the sbml file to be processed.", required=True)

    simulate = subparsers.add_parser("simulate")

    simulate.add_argument("-ts", "--times", help="time points.", nargs="+",type=float, required=True)
    simulate.add_argument("-o", "--output-file", help="where to store the differential equations system.", required=True)
    simulate.add_argument("-t0", "--initial-time", help="initial time point.",type=float, default=0)
    simulate.add_argument("-dt0", "--initial-time-step", help="initial time step.",type=float, default=0.1)


    infos = subparsers.add_parser("infos")
    infos.add_argument("-pv", "--parameters-values", help="list parameters values.", action="store_true")

    adjust = subparsers.add_parser("adjust")
    adjust.add_argument("-ts", "--times", help="time points.", nargs="+",type=float, required=True)
    adjust.add_argument("-t0", "--initial-time", help="initial time point.",type=float, default=0)
    adjust.add_argument("-dt0", "--initial-time-step", help="initial time step.",type=float, default=0.1)
    adjust.add_argument("-lr", "--learning-rate", help="learning rate.",type=float, default=0.01)
    adjust.add_argument("-n", "--num-iterations", help="number of iterations.",type=int, default=100)




    args = parser.parse_args()

    system = getSystem(args.input_file)
    if args.command == "simulate":
        run(args.input_file, args.output_file, args.times, t0=args.initial_time, dt0=args.initial_time_step)
    if args.command == "infos":
        getInfos(system)
    if args.command == "adjust":
        optimize(system, args.times, args.initial_time, args.initial_time_step, args.learning_rate, args.num_iterations)




