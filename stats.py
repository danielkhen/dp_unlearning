import loader
import static
import torch

def print_model(model, layer_num):
    layers = list(model.children())

    for i in range(layer_num):
        layer = layers[i]

        for param in layer.parameters():
            print(param.data)

model_name = 'model'
device = torch.device("cpu")
model = loader.load_model(static.ARCHITECTURE, static.CLASSES_NUM, device, weights_from=f'weights/{model_name}.pth')

layer_num = len(list(model.children()))
param_num = len(list(model.parameters()))

print(
    f"Model name: {model_name} "
    f"Layers num: {layer_num} "
    f"Params num: {param_num} "
)

print_model(model, 1)