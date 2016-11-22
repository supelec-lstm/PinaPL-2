//
//  main.cpp
//  PinaPL
//
//  Created by Maxime on 11/19/16.
//  Copyright © 2016 PinaPL. All rights reserved.
//

#include <iostream>
#include <fstream>

#include "PinaPLDefines.hpp"
#include "NeuralNetworkDefines.hpp"
#include "Functions.hpp"
#include "Neuron.hpp"
#include "NeuralConnection.hpp"
#include "NeuralNetwork.hpp"
#include "idxParser.hpp"


using namespace std;

void testXOR();
void testMNIST();


int main() {
    srand(static_cast<unsigned int>(time(NULL)));

    testMNIST();
    
    return 0;
}

void testXOR() {
/* Build the network */
    NeuralNetwork network = NeuralNetwork();
    network.setDefaultActivationFunction(activationFunctionSigmoid);
    network.setLearningRate(0.3);
    
    network.addNeuron();
    network.addNeuron();
    network.addNeuron();
    network.addNeuron();
    network.addNeuron();
    
    network.addInput(network.neuronAtIndex(0));
    network.addInput(network.neuronAtIndex(1));
    network.addOutput(network.neuronAtIndex(4));
    
    network.addConnection(network.neuronAtIndex(0), network.neuronAtIndex(2));
    network.addConnection(network.neuronAtIndex(0), network.neuronAtIndex(3));
    
    network.addConnection(network.neuronAtIndex(1), network.neuronAtIndex(2));
    network.addConnection(network.neuronAtIndex(1), network.neuronAtIndex(3));
    
    network.addConnection(network.neuronAtIndex(2), network.neuronAtIndex(4));
    network.addConnection(network.neuronAtIndex(3), network.neuronAtIndex(4));
    
    network.neuronAtIndex(0)->setActivationFunction(activationFunctionLinear);
    network.neuronAtIndex(1)->setActivationFunction(activationFunctionLinear);
    
    network.getInputAtIndex(0)->setValue(0);
    network.getInputAtIndex(1)->setValue(0);
    
    
/* Learn XOR */
    vector<double> expectedOutput = vector<double>(1);
    expectedOutput[0] = 1;
    
    for (UInteger i = 0; i < 100000; i++) {
        UInteger a = static_cast<long>(static_cast<double>(rand()) / RAND_MAX * 2.0);
        UInteger b = static_cast<long>(static_cast<double>(rand()) / RAND_MAX * 2.0);
        
        network.getInputAtIndex(0)->setValue(a);
        network.getInputAtIndex(1)->setValue(b);
        expectedOutput[0] = 1 - (a * b + (1 - a) * (1 - b));
        
        network.backpropagation(expectedOutput);
        network.computeWeightVariations();
        
        if (i % 100 == 0)
            network.setNewWeights();
    }
    
/* Tests */
    network.getInputAtIndex(0)->setValue(0);
    network.getInputAtIndex(1)->setValue(0);
    network.computeOutputs();
    cout << network.getOutputValues()[0] << endl;
    network.getInputAtIndex(0)->setValue(1);
    network.getInputAtIndex(1)->setValue(0);
    network.computeOutputs();
    cout << network.getOutputValues()[0] << endl;
    network.getInputAtIndex(0)->setValue(0);
    network.getInputAtIndex(1)->setValue(1);
    network.computeOutputs();
    cout << network.getOutputValues()[0] << endl;
    network.getInputAtIndex(0)->setValue(1);
    network.getInputAtIndex(1)->setValue(1);
    network.computeOutputs();
    cout << network.getOutputValues()[0] << endl;
    
/* MATLAB Plot */
    double x = -0.5, y = -0.5;
    ofstream values;
    values.open("./PinaPL.m");
    values << "a = [";
    
    while (x < 1.5) {
        while (y < 1.5) {
            network.getInputAtIndex(0)->setValue(x);
            network.getInputAtIndex(1)->setValue(y);
            network.computeOutputs();
            values << network.getOutputValues()[0] << " ";
            y += 0.02;
        }
        y = -0.5;
        x += 0.02;
        values << "; ";
    }
    values << "];\n";
    values << "image(flipud(a), 'CDataMapping', 'scaled');";
    values.close();
}

void testMNIST() {
/* Import data */
    IdxParser parser = IdxParser();
    vector<vector<double>> learningData = parser.importMNISTImages("./Ressources/train-images-idx3-ubyte.gz");
    vector<vector<double>> testingData = parser.importMNISTImages("./Ressources/t10k-images-idx3-ubyte.gz");
    vector<Integer> learningDataOutput = parser.importMNISTLabels("./Ressources/train-labels-idx1-ubyte.gz");
    vector<Integer> testingDataOutput = parser.importMNISTLabels("./Ressources/t10k-labels-idx1-ubyte.gz");

    NeuralNetwork network = NeuralNetwork();
    network.setDefaultActivationFunction(activationFunctionSigmoid);
    network.setLearningRate(0.5);
    
    for (UInteger i = 0; i < 784; i++)
        network.addNeuron();
    for (UInteger i = 0; i < 10; i++)
        network.addNeuron();

    for (UInteger i = 0; i < network.neuronCount(); i++) {
        if (i < 784) {
            network.addInput(network.neuronAtIndex(i));
            network.neuronAtIndex(i)->setActivationFunction(activationFunctionLinear);
        }
        else
            network.addOutput(network.neuronAtIndex(i));
    }
    
    for (UInteger i = 0; i < network.inputCount(); i++) {
        for (UInteger j = 0; j < network.outputCount(); j++) {
            network.addConnection(network.neuronAtIndex(i), network.neuronAtIndex(network.inputCount() + j));
        }
    }
    
/* Learn characters */
    vector<double> expectedOutput = vector<double>(10);
    for (UInteger i = 0; i < 10; i++)
        expectedOutput[i] = 0;
    
    UInteger counter = 0;
    for (UInteger k = 0; k < 10; k++) {
        for (UInteger i = 0; i < learningData.size(); i++) {
//        for (UInteger i = 0; i < 100; i++) {
            for (UInteger j = 0; j < learningData[i].size(); j++) {
                network.getInputAtIndex(j)->setValue(learningData[i][j]);
            }
            if (i > 0)
                expectedOutput[learningDataOutput[i - 1]] = 0;
            expectedOutput[learningDataOutput[i]] = 1;
            
            network.backpropagation(expectedOutput);
            network.computeWeightVariations();
            
            if (counter % 10 == 0)
                network.setNewWeights();
            counter++;
        }
    }
    
/* Test learning data */
    vector<double> output = vector<double>(10);
    Integer maxIndex = 0;
    
    double correct = 0.0;
    for (UInteger i = 0; i < learningData.size(); i++) {
//    for (UInteger i = 0; i < 50; i++) {
        for (UInteger j = 0; j < learningData[i].size(); j++)
            network.getInputAtIndex(j)->setValue(learningData[i][j]);
        
        network.computeOutputs();
        
        output = network.getOutputValues();
        for (UInteger j = 0; j < output.size(); j++) {
            if (output[maxIndex] < output[j])
                maxIndex = j;
        }
        
        if (maxIndex == learningDataOutput[i])
            correct += 1.0;
        cout << "Expected: " << learningDataOutput[i] << ", got: " << maxIndex << ", values: ";
        for (UInteger j = 0; j < output.size(); j++)
            cout << output[j] << " ";
        cout << endl;
    }
    cout << endl;
    cout << correct << " " << correct / learningDataOutput.size() << endl;
    cout << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl;

/* Test testing data */
    correct = 0.0;
    for (UInteger i = 0; i < testingData.size(); i++) {
//    for (UInteger i = 0; i < 50; i++) {
        for (UInteger j = 0; j < testingData[i].size(); j++)
            network.getInputAtIndex(j)->setValue(testingData[i][j]);
        
        network.computeOutputs();
        
        output = network.getOutputValues();
        for (UInteger j = 0; j < output.size(); j++) {
            if (output[maxIndex] < output[j])
                maxIndex = j;
        }
        
        if (maxIndex == testingDataOutput[i])
            correct += 1.0;
        cout << "Expected: " << testingDataOutput[i] << ", got: " << maxIndex << ", values: ";
        for (UInteger j = 0; j < output.size(); j++)
            cout << output[j] << " ";
        cout << endl;
    }
    cout << endl;
    cout << correct << " " << correct / testingData.size() << endl;
}
