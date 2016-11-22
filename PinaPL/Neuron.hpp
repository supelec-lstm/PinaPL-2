//
//  Neuron.hpp
//  PinaPL
//
//  Created by Maxime on 11/19/16.
//  Copyright © 2016 PinaPL. All rights reserved.
//

#ifndef Neuron_hpp
#define Neuron_hpp

#include <stdio.h>
#include <vector>


class NeuralConnection;
class NeuralNetwork;

class Neuron {
    std::vector<NeuralConnection*> inputs;
    std::vector<NeuralConnection*> outputs;
    double output;
    double error;

    CompositionFunction compositionFunction;
    ActivationFunction activationFunction;
    
public:
    Neuron();
    
    double composedInput();
    
    double getOutput();
    void computeOutput();
    double getError();
    void computeError();
    void resetError();

    size_t getInputCount();
    NeuralConnection* getInputAtIndex(UInteger index);
    size_t getOutputCount();
    NeuralConnection* getOutputAtIndex(UInteger index);
    CompositionFunction getCompositionFunction();
    ActivationFunction getActivationFunction();
    
    void addInput(NeuralConnection* anInput);
    void removeInput(NeuralConnection* anInput);
    void addOutput(NeuralConnection* anOutput);
    void removeOutput(NeuralConnection* anOutput);
    void setCompositionFunction(CompositionFunction aFunction);
    void setActivationFunction(ActivationFunction aFunction);
};

#endif /* Neuron_hpp */
