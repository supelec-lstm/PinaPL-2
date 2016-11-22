//
//  NeuralNetwork.hpp
//  PinaPL
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>

#include "Neuron.hpp"
#include "NeuralConnection.hpp"


class NeuralNetwork {
    std::vector<Neuron*> neurons;
    std::vector<NeuralConnection*> connections;
    std::vector<NeuralConnection*> inputs;
    std::vector<NeuralConnection*> outputs;
    
    CompositionFunction defaultCompositionFunction;
    ActivationFunction defaultActivationFunction;
    CostFunction defaultCostFunction;
    
    double learningRate;
    
public:
    NeuralNetwork();
    NeuralNetwork (const NeuralNetwork &network);
    ~NeuralNetwork();
    
    void computeOutputs();
    double totalError(std::vector<double> expectedOutputs);
    
    void backpropagation(std::vector<double> expectedOutputs);
    void computeWeightVariations();
    void setNewWeights();
    
    size_t neuronCount();
    Neuron* neuronAtIndex(NeuronIndex index);
    NeuronIndex indexOfNeuron(Neuron *neuron);
    Neuron* addNeuron();
    void removeNeuron(Neuron *neuron);

    size_t connectionCount();
    NeuralConnection* connectionAtIndex(NeuralConnectionIndex index);
    NeuralConnection* addConnection(Neuron* from, Neuron* to, double weight = 0, NeuralConnectionType type = NeuralConnectionTypeRegular);
    void removeConnection(NeuralConnection *connection);

    size_t inputCount();
    NeuralConnection* getInputAtIndex(NeuronIndex index);
    NeuralConnection* addInput(Neuron* neuron);
    void removeInput(NeuralConnection* input);
    
    size_t outputCount();
    Neuron* getOutputNeuronAtIndex(NeuronIndex index);
    std::vector<double> getOutputValues();
    void addOutput(Neuron* neuron);
    void removeOutputNeuron(Neuron* neuron);
    
    CompositionFunction getDefaultCompositionFunction();
    ActivationFunction getDefaultActivationFunction();
    CostFunction getDefaultCostFunction();
    void setDefaultCompositionFunction(CompositionFunction aFunction);
    void setDefaultActivationFunction(ActivationFunction aFunction);
    void setDefaultCostFunction(CostFunction aFunction);

    double getLearningRate();
    void setLearningRate(double newValue);
};

#endif /* NeuralNetwork_hpp */
