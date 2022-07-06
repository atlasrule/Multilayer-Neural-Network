#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <stdexcept>
using namespace std;

random_device rd;
mt19937 e2(rd());
uniform_real_distribution<float> get_random_weight(0, 1);

class Neuron {

  private:

    void initializeRandomWeights(int n_weights) {
      for (unsigned int i=0; i<n_weights; i++ ) {
        float random_weight = get_random_weight(e2);
        weights.push_back(random_weight);
      }
    }

    float sigmoid(float x, float clamp=1000) {
      x = min(clamp, max(-clamp, x));
      //cout << "\nsigmoid: " << (1 / (1 + exp(-x)));
      return 1 / (1 + exp(-x));
    }

    float derivative(float x) {
      return x * (1.0 - x);
    }

    float neuronOutput() {
      float sum = 0.0;
      for (unsigned int i=0; i<this->input_signals.size(); i++) {
        sum += this->input_signals[i] * this->weights[i];
      }
      return sigmoid(sum);
    }

  public:

    float sigma;
    float delta;
    float bias;
    int n_weights;
    vector<float> weights;
    vector<float> input_signals;

    Neuron(int n_weights, vector<float> weights = {}) {
      this->bias = 0;
      this->n_weights = n_weights;
      if (weights.size() == 0) { this->initializeRandomWeights(n_weights); }
      else { this->weights = weights; }
      this->n_weights = n_weights;
    }

    float getSigma(vector<float> input_signals) {
      this->input_signals = input_signals;
      this->sigma = neuronOutput();
      return this->sigma;
    }
};


class MultilayerNeuralNetwork {

  private:

    vector<int> layer_sizes;
    float learning_rate;

    float derivative(float x) {
      return x * (1.0 - x);
    }

    float error(float actual, float predicted) {
      return actual-predicted;
    }

    float output_delta(float learning_rate, float target_delta, float actual, float predicted) {
      
      return learning_rate * derivative(predicted) * error(actual, predicted) * predicted;
    }
    
    float output_bias(float learning_rate, float target_delta, float actual, float predicted) {
      
      return learning_rate * derivative(predicted) * error(actual, predicted) * predicted;
    }

    float new_weight(float learning_rate, float old_weight, float source_signal, float target_delta) {
      
      return old_weight + learning_rate * derivative(source_signal) * target_delta * source_signal;
    }

    float new_bias(float learning_rate, float old_weight, float source_signal, float target_delta) {

      return old_bias + learning_rate * derivative(source_signal) * target_delta * source_signal;
    }


  public:

    vector<vector<Neuron>> layers;

    float learning_rate;

    MultilayerNeuralNetwork(vector<int> layer_sizes, vector<float> actuals, vector<float> predicteds, float learning_rate = 0.0001) {
      this->learning_rate = learning_rate;
      this->layer_sizes = layer_sizes;
      int n_layers = layer_sizes.size(); 

      vector<Neuron> layerNeurons;
      // Initialize input neurons with zero weights;
      int n_input_neurons = layer_sizes[0];
      for (int i=0; i< n_input_neurons; i++) {
        Neuron neuron = *new Neuron((0), {});
        layerNeurons.push_back( neuron );
      }
      this->layers.push_back(layerNeurons);
      layerNeurons.clear();
      

      for (unsigned int layer_i=1; layer_i<n_layers; layer_i++) {
        
        int n_neurons =layer_sizes[layer_i];
        int n_weights = layer_sizes[layer_i-1];
        
        for (int i=0; i< n_neurons; i++) {
          Neuron neuron = *new Neuron((n_weights), {});
          layerNeurons.push_back( neuron );
        }
        
        this->layers.push_back(layerNeurons);
        layerNeurons.clear();
      }
    }
    
    vector<float> feedforward(vector<float> input_signals) {
      if (input_signals.size() != layers[0].size() || layers[0].size() != layer_sizes[0]) {
        invalid_argument("Inconsisent input signals size with input layer.");
        cout << "Inconsisent input signals size with input layer.";
        exit(0);
      }
      for (unsigned int neuron_i=0; neuron_i<layers[0].size(); neuron_i++) {
        layers[0][neuron_i].sigma = input_signals[neuron_i];
        //cout << "\nsigmoid::" << layers[0][neuron_i].sigma;
      }
      
      for (unsigned int layer_i=1; layer_i<layers.size(); layer_i++) {
        
        for (unsigned int neuron_i=0; neuron_i<layers[layer_i-1].size(); neuron_i++) {
          input_signals.push_back(layers[layer_i-1][neuron_i].sigma);
        }
        
        for (unsigned int neuron_i=0; neuron_i<layers[layer_i].size(); neuron_i++) {
          layers[layer_i][neuron_i].getSigma(input_signals);
        }
        
        input_signals.clear();
      }
      int n_layers = layer_sizes.size();
      int n_output_neurons = this->layer_sizes[this->layer_sizes.size()-1];
      vector<float> output_signals;
      for (int neuron_i=0; neuron_i < n_output_neurons; neuron_i++) {
        output_signals.push_back(layers[n_layers-1][neuron_i].sigma);
      }
      
      return output_signals;
    }

    void backpropagate( vector<float> actuals, vector<float> predicteds) {
      
      int n_layers = this->layers.size();

      // Calculate output layer errors.
      int output_layer_i = layers[n_layers-1].size();
      for (int neuron_i=0; neuron_i<output_layer_i; neuron_i++) {
        Neuron neuron = layers[output_layer_i][neuron_i];
        float output_signal = neuron.sigma;
        neuron.delta = output_delta(this->learning_rate, output_signal, this->actuals[neuron_i], this->predicteds[neuron_i]);
      }

      // Update all weights.
      for (int layer_i = n_layers-1; layer_i > 0; layer_i--) {
        for (int neuron_i=0; neuron_i<layers[layer_i].size(); neuron_i++) {
          Neuron neuron = layers[layer_i][neuron_i];
          Neuron source_neuron = layers[layer_i-1][neuron_i];
          for(int weight_i=0; weight_i<neuron.n_weights; weight_i++) {
            float old_weight = neuron.weights[weight_i];
            float source_signal = source_neuron.sigma;
            float target_delta = neuron.delta;
            neuron.weights[weight_i] = new_weight(learning_rate, old_weight, source_signal, target_delta);
          }
        }
      }
    }
};


int main() {
  vector<int> layer_sizes = {2,2,1};

  vector<float> input_signals = {1, 1};
  
  MultilayerNeuralNetwork mnn(layer_sizes);
    
  vector<float> actuals = {1};
  vector<float> predicteds = mnn.feedforward(input_signals);

  mnn.backpropagate(actuals, predicteds);
}