# Muslum Berkay Yılmaz
# No: 2018280056

from random import uniform
from math import exp


def random_list(n):
  return [uniform(-1,1) for random_weight in range(n)]


def initialize_random_weights(weights, layer_sizes):
  for layer_idx in range(1, len(layer_sizes)):
    weight_count_per_neuron = layer_sizes[layer_idx-1]
    weights.append([])
    for recieving_neuron in range(layer_sizes[layer_idx]):
      weights[layer_idx-1].append(random_list(weight_count_per_neuron))


def neuron_output(X, W, bias=0):
  return sigmoid(sum([x * w for x, w in zip(X, W)]) + bias)


def sigmoid(x, clamp=1000):
  x = min(clamp, max(-clamp, x))
  return 1 / (1 + exp(-x))
  

def derivative(x):
  return x * (1.0 - x)


def error(expected, predicted):
  return (expected-predicted)**2


def total_error(expected_outputs, predicted_outputs):
  total_error = 0
  for i in range(len(predicted_outputs)):
    total_error += error(expected_outputs[i], predicted_outputs[i])
  return total_error


def create_network(train_inputs, layer_sizes):
  if len(train_inputs[0]) != layer_sizes[0]:
    print(len(train_inputs))
    raise ValueError("Create Network: train_inputs must be equal to input layer size.")
  
  for neuron_count in layer_sizes:
    signals.append([0 for neuron in range(neuron_count)])

  initialize_random_weights(weights, layer_sizes)
  for train_input in train_inputs:
    for input_signal_idx in range(len(train_input)):
      signals[0][input_signal_idx] = train_input[input_signal_idx]


def feedforward(layer_sizes):
  for layer_idx in range(len(layer_sizes)-1):
    for recieving_neuron_idx in range(layer_sizes[layer_idx+1]):
      signals[layer_idx+1][recieving_neuron_idx] = neuron_output(signals[layer_idx], weights[layer_idx][recieving_neuron_idx])


def backpropagate(layer_sizes, expected_outputs):
  for layer_idx in reversed(range(len(layer_sizes))):
    errors = []

    if layer_idx == len(layer_sizes)-1:
      for neuron_idx in range(layer_sizes[layer_idx]):
        errors.append(signals[layer_idx][neuron_idx] - expected_outputs[neuron_idx])

    elif layer_idx != len(layer_sizes)-1:
      for neuron_idx in range(layer_sizes[layer_idx]):
        error = 0.0
        for recieving_neuron_idx in range(layer_sizes[layer_idx+1]):
          error += weights[layer_idx][recieving_neuron_idx][neuron_idx] * deltas[layer_idx+1][recieving_neuron_idx]
        errors.append(error)

    for neuron_idx in range(layer_sizes[layer_idx]):
      deltas[layer_idx][neuron_idx] = errors[neuron_idx] * derivative(signals[layer_idx][neuron_idx])


def update_weights(layer_sizes, inputs, learning_rate):
  for layer_idx in range(len(layer_sizes)):
    if layer_idx != 0:
      inputs = [signals[layer_idx-1][neuron_idx] for neuron_idx in range(layer_sizes[layer_idx-1])]
      for neuron_idx in range(layer_sizes[layer_idx]):
        for input_idx in range(len(inputs)):
          weights[layer_idx][1][neuron_idx] -= learning_rate * deltas[layer_idx][neuron_idx] * inputs[input_idx]
        weights[layer_idx][-1][neuron_idx] -= learning_rate * deltas[layer_idx][neuron_idx]



def train(layer_sizes, epochs, train_inputs, train_outputs):

    signals = [[0 for neuron in range(layer_size)] for layer_size in layer_sizes]
    weights = []
    deltas = [[0 for neuron in range(layer_size)] for layer_size in layer_sizes]

    create_network(train_inputs=train_inputs, train_outputs=train_outputs, layer_sizes=layer_sizes)

    # Train loop
    for epoch in range(epochs):
        error_sum = 0
        for train_idx in train_inputs:
            feedforward(layer_sizes=layer_sizes)
            outputs = signals[-1] #Last layer of signals are the outputs of the MLP
            error_sum += total_error(expected_outputs=train_outputs, predicted_outputs=outputs)
            backpropagate(layer_sizes=layer_sizes, expected_outputs=train_outputs)
            update_weights(layer_sizes, train_inputs, learning_rate=0.3)

        print('Epoch: ', epoch)
        print('Neuron signals:\n', signals)
        print('Weight deltas:\n', deltas)
        print('Weight deltas:\n'