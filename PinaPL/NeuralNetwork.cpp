//
//  NeuralNetwork.cpp
//  PinaPL
//
//  Created by Maxime on 11/19/16.
//  Copyright © 2016 PinaPL. All rights reserved.
//

#include <random>
#include <map>
#include <queue>

#include "PinaPLDefines.hpp"
#include "NeuralNetworkDefines.hpp"
#include "Functions.hpp"
#include "Neuron.hpp"
#include "NeuralConnection.hpp"

#include "NeuralNetwork.hpp"


using namespace std;

NeuralNetwork::NeuralNetwork() {
    neurons = vector<Neuron*>();
    connections = vector<NeuralConnection*>();
    inputs = vector<NeuralConnection*>();
    outputs = vector<NeuralConnection*>();
    
    defaultCompositionFunction = compositionFunctionSum;
    defaultActivationFunction = activationFunctionSigmoid;
    defaultCostFunction = costFunctionQuadratic;
    
    learningRate = 1.0;
}

NeuralNetwork::~NeuralNetwork() {
    for (UInteger i = 0; i < neurons.size(); i++)
        delete neurons[i];
    for (UInteger i = 0; i < connections.size(); i++) {
        delete connections[i];
    }
}

#pragma mark - Compute operations

void NeuralNetwork::computeOutputs() {
    map<Neuron*, bool> queuedNeurons = map<Neuron*, bool>();
    queue<Neuron*> neuronQueue = queue<Neuron*>();

    Neuron *neuron = NULL;
    Neuron *next = NULL;

    for (UInteger i = 0; i < neurons.size(); i++)
        queuedNeurons[neurons[i]] = false;
    for (UInteger i = 0; i < inputs.size(); i++) {
        neuron = inputs[i]->getToNeuron();
        neuronQueue.push(neuron);
        queuedNeurons[neuron] = true;
    }
    
    while (!neuronQueue.empty()) {
        neuron = neuronQueue.front();
        neuronQueue.pop();
        queuedNeurons[neuron] = false;
        neuron->computeOutput();
        for (UInteger i = 0; i < neuron->getOutputCount(); i++) {
            next = neuron->getOutputAtIndex(i)->getToNeuron();
            if (next && !queuedNeurons[next]) {
                neuronQueue.push(next);
                queuedNeurons[next] = true;
            }
        }
    }
}

double NeuralNetwork::totalError(vector<double> expectedOutputs) {
    vector<double> outputs = this->getOutputValues();
    return defaultCostFunction.main(&outputs, &expectedOutputs);
}

void NeuralNetwork::backpropagation(vector<double> expectedOutputs) {
    this->computeOutputs();
    
    map<Neuron*, bool> queuedNeurons = map<Neuron*, bool>();
    queue<Neuron*> neuronQueue = queue<Neuron*>();
    
    Neuron *neuron = NULL;
    Neuron *previous = NULL;

    for (UInteger i = 0; i < neurons.size(); i++)
        queuedNeurons[neurons[i]] = false;
    for (UInteger i = 0; i < outputs.size(); i++) {
        neuron = outputs[i]->getFromNeuron();
        neuronQueue.push(neuron);
        queuedNeurons[neuron] = true;
    }
    
    for (UInteger i = 0; i < outputs.size(); i++)
        outputs[i]->setExpectedValue(expectedOutputs[i]);
    
    while (!neuronQueue.empty()) {
        neuron = neuronQueue.front();
        neuronQueue.pop();
        queuedNeurons[neuron] = false;
        
        neuron->computeError();

        for (UInteger i = 0; i < neuron->getInputCount(); i++) {
            previous = neuron->getInputAtIndex(i)->getFromNeuron();
            if (previous && !queuedNeurons[previous]) {
                neuronQueue.push(previous);
                queuedNeurons[previous] = true;
            }
        }
    }

}

void NeuralNetwork::computeWeightVariations() {
    for (UInteger i = 0; i < connections.size(); i++)
        connections[i]->addCurrentWeightVariation();
}

void NeuralNetwork::setNewWeights() {
    for (UInteger i = 0; i < connections.size(); i++)
        connections[i]->computeNewWeight(learningRate);
}

#pragma mark - Neurons

size_t NeuralNetwork::neuronCount() {
    return neurons.size();
}

Neuron* NeuralNetwork::neuronAtIndex(NeuronIndex index) {
    return neurons[index];
}

NeuronIndex NeuralNetwork::indexOfNeuron(Neuron *neuron) {
    for (UInteger i = 0; i< neurons.size(); i++) {
        if (neuron == neurons[i])
            return i;
    }
    return NotFound;
}

Neuron* NeuralNetwork::addNeuron() {
    Neuron *neuron = new Neuron();
    neuron->setCompositionFunction(defaultCompositionFunction);
    neuron->setActivationFunction(defaultActivationFunction);
    neurons.push_back(neuron);
    this->addConnection(NULL, neuron, 0.0, NeuralConnectionTypeBias);
    return neuron;
}

void NeuralNetwork::removeNeuron(Neuron *neuron) {
    for (UInteger i = 0; i< neurons.size(); i++) {
        if (neuron == neurons[i]) {
            neurons.erase(neurons.begin() + i);
            return;
        }
    }
}

#pragma mark - Connections

size_t NeuralNetwork::connectionCount() {
    return connections.size();
}

NeuralConnection* NeuralNetwork::connectionAtIndex(NeuralConnectionIndex index) {
    return connections[index];
}

NeuralConnection* NeuralNetwork::addConnection(Neuron* from, Neuron* to, double weight, NeuralConnectionType type) {
    assert(from || to);
    if (type == NeuralConnectionTypeRegular) {
        for (UInteger i = 0; i < from->getOutputCount(); i++) {
            if (from->getOutputAtIndex(i)->getToNeuron() == to) {
                assert(false);
            }
        }
    }
    
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
    while (weight == 0)
        weight = (static_cast<double>(rand()) / RAND_MAX) * 2 - 1;
#pragma clang diagnostic pop
    
    NeuralConnection *connection = new NeuralConnection(type);
    connection->setFromNeuron(from);
    connection->setToNeuron(to);
    connection->setWeight(weight);

    if (!(type & NeuralConnectionTypeIn))
        from->addOutput(connection);
    if (!(type & NeuralConnectionTypeOut))
        to->addInput(connection);

    connections.push_back(connection);
    return connection;
}

void NeuralNetwork::removeConnection(NeuralConnection *connection) {
    for (UInteger i = 0; i< connections.size(); i++) {
        if (connection == connections[i]) {
            connections.erase(connections.begin() + i);
            return;
        }
    }
}

#pragma mark - Inputs

size_t NeuralNetwork::inputCount() {
    return inputs.size();
}

NeuralConnection* NeuralNetwork::getInputAtIndex(NeuronIndex index) {
    return inputs[index];
}

NeuralConnection* NeuralNetwork::addInput(Neuron* neuron) {
    for (UInteger i = 0; i< neuron->getInputCount(); i++) {
        if (neuron->getInputAtIndex(i)->getType() == NeuralConnectionTypeBias) {
            NeuralConnection *input = neuron->getInputAtIndex(i);
            input->setType(NeuralConnectionTypeInput);
            inputs.push_back(input);
            return input;
        }
        if (neuron->getInputAtIndex(i)->getType() == NeuralConnectionTypeInput)
            return neuron->getInputAtIndex(i);
    }
    return NULL;
}

void NeuralNetwork::removeInput(NeuralConnection* input) {
    for (UInteger i = 0; i< inputs.size(); i++) {
        if (input == inputs[i]) {
            input->setType(NeuralConnectionTypeBias);
            inputs.erase(inputs.begin() + i);
            return;
        }
    }
}

#pragma mark - Outputs

size_t NeuralNetwork::outputCount() {
    return outputs.size();
}

Neuron* NeuralNetwork::getOutputNeuronAtIndex(NeuronIndex index) {
    return outputs[index]->getFromNeuron();
}

vector<double> NeuralNetwork::getOutputValues() {
    vector<double> outputValues = vector<double>(outputs.size());
    for (UInteger i = 0; i < outputValues.size(); i++) {
        outputValues[i] = outputs[i]->getValue();
    }
    return outputValues;
}

void NeuralNetwork::addOutput(Neuron* neuron) {
    for (UInteger i = 0; i< neuron->getOutputCount(); i++) {
        if (neuron->getOutputAtIndex(i)->getType() == NeuralConnectionTypeOutput) {
            return;
        }
    }
    
    NeuralConnection *output = this->addConnection(neuron, NULL, 1.0, NeuralConnectionTypeOutput);
    output->setCostFunction(defaultCostFunction);
    outputs.push_back(output);
}

void NeuralNetwork::removeOutputNeuron(Neuron* neuron) {
    for (UInteger i = 0; i< outputs.size(); i++) {
        if (neuron == outputs[i]->getFromNeuron()) {
            outputs.erase(outputs.begin() + i);
            return;
        }
    }
}

#pragma mark - Compostion, activation and cost functions

CompositionFunction NeuralNetwork::getDefaultCompositionFunction() {
    return defaultCompositionFunction;
}

ActivationFunction NeuralNetwork::getDefaultActivationFunction() {
    return defaultActivationFunction;
}

CostFunction NeuralNetwork::getDefaultCostFunction() {
    return defaultCostFunction;
}

void NeuralNetwork::setDefaultCompositionFunction(CompositionFunction aFunction) {
    defaultCompositionFunction = aFunction;
}

void NeuralNetwork::setDefaultActivationFunction(ActivationFunction aFunction) {
    defaultActivationFunction = aFunction;
}

void NeuralNetwork::setDefaultCostFunction(CostFunction aFunction) {
    defaultCostFunction = aFunction;
}

#pragma mark - Learning rate

double NeuralNetwork::getLearningRate() {
    return learningRate;
}

void NeuralNetwork::setLearningRate(double newValue) {
    learningRate = newValue;
}
