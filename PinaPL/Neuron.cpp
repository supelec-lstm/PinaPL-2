//
//  Neuron.cpp
//  PinaPL
//
//  Created by Maxime on 11/19/16.
//  Copyright © 2016 PinaPL. All rights reserved.
//


#include "PinaPLDefines.hpp"
#include "NeuralNetworkDefines.hpp"
#include "Functions.hpp"
#include "NeuralConnection.hpp"
#include "NeuralNetwork.hpp"

#include "Neuron.hpp"


using namespace std;

Neuron::Neuron() {
    inputs = vector<NeuralConnection*>();
    outputs = vector<NeuralConnection*>();
    output = 0;
    error = 0;
    
    compositionFunction = compositionFunctionSum;
    activationFunction = activationFunctionSigmoid;
}

double Neuron::composedInput() {
    return compositionFunction.main(&inputs);
}

double Neuron::getOutput() {
    return output;
}

void Neuron::computeOutput() {
    double composedInputs = this->composedInput();
    output = activationFunction.main(&composedInputs);
    for (UInteger i = 0; i < outputs.size(); i++)
        outputs[i]->setValue(output);
}

double Neuron::getError() {
    return error;
}

void Neuron::computeError() {
    error = 0;
    for (UInteger i = 0; i < outputs.size(); i++)
        error = error + outputs[i]->getError() * outputs[i]->getWeight();
    error = error * activationFunction.derivative(&output);
    
    for (UInteger i = 0; i < inputs.size(); i++) {
        if (inputs[i]->getType() != NeuralConnectionTypeInput)
            inputs[i]->setError(error);
    }
}

void Neuron::resetError() {
    error = 0;
}

#pragma mark - Getters and Setters

size_t Neuron::getInputCount() {
    return inputs.size();
}

NeuralConnection* Neuron::getInputAtIndex(UInteger index) {
    return inputs[index];
}

size_t Neuron::getOutputCount() {
    return outputs.size();
}

NeuralConnection* Neuron::getOutputAtIndex(UInteger index) {
    return outputs[index];
}

CompositionFunction Neuron::getCompositionFunction() {
    return compositionFunction;
}

ActivationFunction Neuron::getActivationFunction() {
    return activationFunction;
}

void Neuron::addInput(NeuralConnection* anInput) {
    assert(anInput->getToNeuron() == this);
    for (UInteger i = 0; i< inputs.size(); i++) {
        if (anInput == inputs[i])
            return;
    }
    inputs.push_back(anInput);
}

void Neuron::removeInput(NeuralConnection* anInput) {
    for (UInteger i = 0; i< inputs.size(); i++) {
        if (anInput == inputs[i]) {
            inputs.erase(inputs.begin() + i);
            return;
        }
    }
}

void Neuron::addOutput(NeuralConnection* anOutput) {
    assert(anOutput->getFromNeuron() == this);
    for (UInteger i = 0; i< outputs.size(); i++) {
        if (anOutput == outputs[i])
            return;
    }
    outputs.push_back(anOutput);
}

void Neuron::removeOutput(NeuralConnection* anOutput) {
    for (UInteger i = 0; i< outputs.size(); i++) {
        if (anOutput == outputs[i]) {
            outputs.erase(outputs.begin() + i);
            return;
        }
    }
}

void Neuron::setCompositionFunction(CompositionFunction aFunction) {
    compositionFunction = aFunction;
}

void Neuron::setActivationFunction(ActivationFunction aFunction) {
    activationFunction = aFunction;
}

