import argparse
import libsbml
from DiffExpr import *

def run(filename, targetname):
    mdl = libsbml.readSBML(filename).getModel()
    system = DiffEqSystem(mdl)
    system.build()
    system.writeDiffEqs(targetname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="the sbml file to be processed.", required=True)
    parser.add_argument("-o", "--output_file", help="where to store the differential equations system.", required=True)
    args = parser.parse_args()
    run(args.input_file, args.output_file)


