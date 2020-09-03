import pickle
import os

def get_layers_dict(model, layers_class_filter):
    layers_dict = {}
    model_type_counter = {}
    for layer in model.modules():
        layer_type = type(layer)
        if layer_type not in model_type_counter:
            model_type_counter[layer_type] = 0
        model_type_counter[layer_type] += 1
        layers_dict[layer] = str(layer_type) + str(model_type_counter[layer_type])
    return layers_dict

class LayerForwardCache(object):
    def __init__(self, layer, layer_input, layer_output):
        self.layer_input = layer_input
        self.layer_output = layer_output
        self.layer = layer

class LayersForwardHandling(object):
    def __init__(self, layers_dict, output_path, gpu):
        self.__layers_dict = layers_dict
        self.__output_path = output_path
        self.__gpu = gpu
        self.__running_index = 0

    def handle_layer_forward(self, layer, layer_input, layer_output):
        layer_forward_cache = LayerForwardCache(self.__layers_dict[layer], layer_input, layer_output)
        pickle.dump(layer_forward_cache, os.path.join(self.__output_path, self.__layers_dict[layer], self.__gpu, self.__running_index))
        del layer_forward_cache
        self.__running_index += 1


def forward_pickle_hook(layer, input, output):
    pickle.dump()

def hook_layers_forward(model):
    for layer in model.modules():
        pass
