# main script
import argparse
from core.circuit import Circuit
from core.trainer import Trainer
from core.utils.customparser import customparser
from core.utils.utils import parse_data, clean_container


parser = argparse.ArgumentParser()
parser.add_argument('config_file_name', help='A yaml file name which contains configurations (e.g. config.yaml)')
arg = parser.parse_args()

custom_parser = customparser(arg.config_file_name)    
parsed_args = custom_parser.parse_custom_args()

circuitarg = parsed_args[0]
optarg = parsed_args[1]
otherarg = parsed_args[2]

clean_container()

t = None

t = Trainer(optarg, circuitarg)

t.train(otherarg.save_to)

parse_data(otherarg.save_to)
