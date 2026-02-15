from .arguments import *
import yaml

# Arguments parser for custom-defined argument dataclasses.

class customparser:

    def __init__(self, file_path):
        self.arg = None
        with open(file_path) as f:
            self.arg = yaml.safe_load(f)
            f.close()

    def parse_custom_args(self):
        return (circuitarguments(num_wires      = self.arg['num_wires'], \
                                num_entangler   = self.arg['num_entangler'], \
                                t2              = self.arg['t2'], \
                                p               = self.arg['p'], \
                                gm_ratio        = self.arg['gm_ratio'], \
                                B               = self.arg['B'], \
                                t               = self.arg['t_obs'], \
                                ps              = self.arg['ps']), \
                optarguments(   opt             = self.arg['opt'], \
                                t_obs           = self.arg['t_obs'], \
                                num_points      = self.arg['num_points'], \
                                steps_per_point = self.arg['steps_per_point'], \
                                patience        = self.arg['patience'], \
                                num_process     = self.arg['num_process']), \
                otherarguments( save_to         = self.arg['save_to']))