//
//  NeuralConnection.hpp
//  PinaPL
//
//  Created by Maxime on 11/19/16.
//  Copyright © 2016 PinaPL. All rights reserved.
//

#ifndef NeuralConnection_hpp
#define NeuralConnection_hpp

#include <stdio.h>

#include "Functions.hpp"


class Neuron;

enum NeuralConnectionType : unsigned long {
    NeuralConnectionTypeRegular = 0,
    NeuralConnectionTypeIO = 1 << 0,
    NeuralConnectionTypeIn = 1 << 2,
    NeuralConnectionTypeOut = 1 << 3,
    NeuralConnectionTypeInput = NeuralConnectionTypeIO | NeuralConnectionTypeIn,
    NeuralConnectionTypeOutput = NeuralConnectionTypeIO | NeuralConnectionTypeOut,
    NeuralConnectionTypeBias = NeuralConnectionTypeIn
};

class NeuralConnection {
    NeuralConnectionType type;
    Neuron* fromNeuron;
    Neuron* toNeuron;
    
    double weight;
    double value;
    
    double error;
    double weightVariation;
    
    CostFunction costFunction;
    double expectedValue;
    
public:
    NeuralConnection(NeuralConnectionType aType = NeuralConnectionTypeRegular);

    double weightedValue();
    void addCurrentWeightVariation();
    void computeNewWeight(double learningRate);
    
    NeuralConnectionType getType();
    Neuron* getFromNeuron();
    Neuron* getToNeuron();
    double getWeight();
    double getValue();
    double getError();
    double getWeightVariation();
    CostFunction getCostFunction();
    double getExpectedValue();

    void setType(NeuralConnectionType newType);
    void setFromNeuron(Neuron* neuron);
    void setToNeuron(Neuron* neuron);
    void setWeight(double newWeight);
    void setValue(double newValue);
    void setError(double newError);
    void setCostFunction(CostFunction aFunction);
    void setExpectedValue(double newValue);
};

#endif /* NeuralConnection_hpp */
