
import ABCParse
import torch, torch_nets, neural_diffeqs


class FunctionFromDescription(ABCParse.ABCParse):
    """formatted func name from func_description"""

    def __init__(self, func_description, func_type):
        
        self.__parse__(locals(), private=['func_description'])
    
    def _simple_stripline(self, line, do_not_strip=["activation", "dropout"]):

        if not any([skip_key in line for skip_key in do_not_strip]):
            return line.replace(")", "").replace("(", "").strip(" ")
        else:
            return line.strip(" ")
    
    @property
    def raw(self):
        return self._func_description
    
    @property
    def raw_by_line(self):
        return self.raw.split("\n")
    
    @property
    def plain_lines(self):
        _plain_lines = [self._simple_stripline(line) for line in self.raw_by_line]
        return [line for line in _plain_lines if len(line) > 0]        
    
    @property
    def header(self):
        return self.plain_lines[0]
    
    # -- kwarg constructors: ------------------------------------------
    def Linear_from_line(self, line):
        split_line = line.split("=")
        in_features = int(split_line[1].split(",")[0])
        out_features = int(split_line[2].split(",")[0])
        bias = split_line[-1]

        return {"in_features": in_features, "out_features": out_features, "bias": bias}
    
    def activation_function_from_line(self, line):
        activation_line = line.split(":")[1].strip(" ").split("(")
        func, arg = activation_line[0], activation_line[1].strip(")")
        key, val = arg.split("=")
        arg = {key: float(val)}
        return {"activation": getattr(torch.nn, func)(**arg)}

    def dropout_function_from_line(self, line):
        return float(line.split("p=")[1].split(",")[0])
    
    @property
    def function_components(self):
        
        if self.func_type != "NeuralSDE":
            return ['mu']
        return ['mu', 'sigma']
    
    def _configure_network_key(self, line):
        """return mu or sigma"""
        if self.func_type == "TorchNet":
            return "mu"
        else:
            return line.split(":")[0]
        
        
    def _reconstruct_network_dict(self):
        
        # TODO: build detection of output bias
        
        NetworkComponents = {}
        for n, line in enumerate(self.plain_lines):
            _func_key = self._configure_network_key(line)
            if (not _func_key in NetworkComponents.keys()) and (_func_key in self.function_components):
                func_key = _func_key
                NetworkComponents[func_key] = {}
            elif "Sequential" in line:
                seq_key = line.split(":")[0]
                NetworkComponents[func_key][seq_key] = []
            elif "Linear" in line:
                NetworkComponents[func_key][seq_key].append(self.Linear_from_line(line))

            elif "activation" in line:
                NetworkComponents[func_key][seq_key].append(self.activation_function_from_line(line))

            elif "dropout" in line:
                NetworkComponents[func_key][seq_key].append(self.dropout_function_from_line(line))
                
        self.NetworkComponents = NetworkComponents 
        
        
    def _compose_network_kwargs(self, network_key):
                
        self.hidden, self.activation, self.dropout = [], [], []
        self._n_hidden = 0
        for key, layer in self.NetworkComponents[network_key].items():            
            if self._n_hidden == 0:
                state_size = layer[0]["in_features"]
            else:
                self.hidden.append(layer[0]["in_features"])
            if len(layer) == 3:
                self.activation.append(layer[2]['activation'])
                self.dropout.append(layer[1])
            elif len(layer) == 2:
                self.activation.append(layer[1]['activation'])
                self.dropout.append(0)
            self._n_hidden += 1
            
        if not self.func_type == "NeuralSDE":
            fmt = ""
        else:
            fmt = "{}_".format(network_key)

        self._FUNC_KWARGS["{}hidden".format(fmt)] = self.hidden
        self._FUNC_KWARGS["{}activation".format(fmt)] = self.activation
        self._FUNC_KWARGS["{}dropout".format(fmt)] = self.dropout
        
        

        if not network_key == "sigma":
            if self.func_type == "TorchNet":
                self._FUNC_KWARGS['in_features'] = self._FUNC_KWARGS['out_features'] = state_size
            else:
                self._FUNC_KWARGS['state_size'] = state_size
        
    
    def _configure_function(self):
        if not hasattr(self, "_function"):
            if self.func_type == "TorchNet":
                pkg = torch_nets
            else:
                pkg = neural_diffeqs
            self._function = getattr(pkg, self.func_type)
    
    @property
    def function(self):
        self._configure_function()
        return self._function
    
    def __call__(self):
        
        
        self._reconstruct_network_dict()
        
        self._FUNC_KWARGS = {}        
        for key in self.function_components:
            self._compose_network_kwargs(key)
            
        return self.function(**self._FUNC_KWARGS)
    
def reconstruct_function(hparams):
    func_from_description = FunctionFromDescription(hparams.func_description, hparams.func_type)
    return func_from_description()
