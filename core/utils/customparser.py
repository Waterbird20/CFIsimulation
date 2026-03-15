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
                                ps              = self.arg['ps'], \
                                fixed_ts        = self.arg.get('fixed_ts', None), \
                                fixed_ps_gamma  = self.arg.get('fixed_ps_gamma', None)), \
                optarguments(   opt                = self.arg['opt'], \
                                maxiter            = self.arg.get('maxiter', 10000), \
                                initial_temp       = self.arg.get('initial_temp', 5230.0), \
                                restart_temp_ratio = self.arg.get('restart_temp_ratio', 2e-5), \
                                visit              = self.arg.get('visit', 2.62), \
                                accept             = self.arg.get('accept', -5.0), \
                                maxfun             = self.arg.get('maxfun', 10000000), \
                                no_local_search    = self.arg.get('no_local_search', False), \
                                t_obs              = self.arg['t_obs'], \
                                num_points         = self.arg['num_points'], \
                                steps_per_point    = self.arg['steps_per_point'], \
                                patience           = self.arg['patience'], \
                                num_process        = self.arg['num_process']), \
                otherarguments( save_to         = self.arg['save_to'],
                                seed            = self.arg.get('seed', 42)))