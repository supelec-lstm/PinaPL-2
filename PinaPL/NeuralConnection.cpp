//
//  NeuralConnection.cpp
//  PinaPL
//
//  Created by Maxime on 11/19/16.
//  Copyright © 2016 PinaPL. All rights reserved.
//

#include "PinaPLDefines.hpp"
#include "NeuralNetworkDefines.hpp"
#include "Functions.hpp"
#include "Neuron.hpp"
#include "NeuralNetwork.hpp"

#include "NeuralConnection.hpp"


NeuralConnection::NeuralConnection(NeuralConnectionType aType) {
    type = aType;
    fromNeuron = NULL;
    toNeuron = NULL;
    
    weight = 1;
    value = (type == NeuralConnectionTypeBias) ? 1.0 : 0.0;
    
    error = 0;
    weightVariation = 0;
    
    costFunction = costFunctionQuadratic;
    expectedValue = 0;
}

double NeuralConnection::weightedValue() {
    return weight * value;
}

void NeuralConnection::addCurrentWeightVariation() {
    weightVariation = weightVariation + error * value;
    error = 0;
}

void NeuralConnection::computeNewWeight(double learningRate) {
    weight = weight - learningRate * weightVariation;
    weightVariation = 0;
}

#pragma mark - Getters and Setters

NeuralConnectionType NeuralConnection::getType() {
    return type;
}

Neuron* NeuralConnection::getFromNeuron() {
    return fromNeuron;
}

Neuron* NeuralConnection::getToNeuron() {
    return toNeuron;
}

double NeuralConnection::getWeight() {
    return weight;
}

double NeuralConnection::getValue() {
    return value;
}

double NeuralConnection::getError() {
    assert(type != NeuralConnectionTypeInput);
    if (type == NeuralConnectionTypeOutput)
        return costFunction.derivative(&value, &expectedValue);
    return error;
}

double NeuralConnection::getWeightVariation() {
    assert(!(type | NeuralConnectionTypeIO));
    return weightVariation;
}

CostFunction NeuralConnection::getCostFunction() {
    assert(type == NeuralConnectionTypeOutput);
    return costFunction;
}

double NeuralConnection::getExpectedValue() {
    assert(type == NeuralConnectionTypeOutput);
    return expectedValue;
}

void NeuralConnection::setType(NeuralConnectionType newType) {
    type = newType;
    error = 0;
    weightVariation = 0;
    expectedValue = 0;
    switch (type) {
        case NeuralConnectionTypeRegular:
            weight = 1;
            value = 0;
            break;
        case NeuralConnectionTypeInput:
            fromNeuron = NULL;
            weight = 1;
            value = 0;
            break;
        case NeuralConnectionTypeOutput:
            toNeuron = NULL;
            weight = 1;
            value = 0;
            break;
        case NeuralConnectionTypeBias:
            fromNeuron = NULL;
            weight = 0;
            value = 1;
            break;
        case NeuralConnectionTypeIO:
//      case NeuralConnectionTypeIn:
        case NeuralConnectionTypeOut:
            assert(false);
            break;
    }
}

void NeuralConnection::setFromNeuron(Neuron* neuron) {
    fromNeuron = neuron;
}

void NeuralConnection::setToNeuron(Neuron* neuron) {
    toNeuron = neuron;
}

void NeuralConnection::setWeight(double newWeight) {
    weight = newWeight;
}

void NeuralConnection::setValue(double newValue) {
    assert(type != NeuralConnectionTypeBias);
    value = newValue;
}

void NeuralConnection::setError(double newError) {
    assert(type != NeuralConnectionTypeOutput);
    error = newError;
}

void NeuralConnection::setCostFunction(CostFunction aFunction) {
    assert(type == NeuralConnectionTypeOutput);
    costFunction = aFunction;
}

void NeuralConnection::setExpectedValue(double newValue) {
    assert(type == NeuralConnectionTypeOutput);
    expectedValue = newValue;
}
