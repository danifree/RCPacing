import argparse

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() +"/data")
sys.path.append(os.getcwd() +"/model")

from model import *
from data.data_loader import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='rcpacing', type=str,
                        help='allocation model')
    parser.add_argument('-d', '--data_path', default='./data/supply_demand_data.txt', type=str,
                        help='path of the data file')

    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = get_args()

    supply_file, demand_file = [args.data_path] * 2
    supply = Supply(supply_file)
    demand = Demand(demand_file)
    edge = Edge(supply_file, demand, supply)

    model = None
    if args.model == 'rcpacing':
        model = RCPacing
    elif args.model == 'dmd':
        model = DMD
    elif args.model == 'smart':
        model = SmartPacing
    elif args.model == 'auaf':
        model = AUAF
    elif args.model == 'pdoa':
        model = PDOA
    else:
        raise ValueError(f'undefined model: {args.model}')
        
    op = model(supply, demand, edge)
    op.alloc_all(100)