from read_png import get_data
from mlp import train


train_data = get_data('train')

train(layer_sizes=[784,80,30,10],
                   epochs = 1000,
                   train_inputs = train_data.inputs,
                   train_outputs = train_data.outputs)
