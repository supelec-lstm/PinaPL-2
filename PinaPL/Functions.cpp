//
//  mathFunctions.cpp
//  PinaPL
//

#include <cmath>

#include "PinaPLDefines.hpp"
#include "NeuralNetworkDefines.hpp"
#include "NeuralConnection.hpp"
#include "Functions.hpp"

#pragma clang diagnostic ignored "-Wunused-parameter"


using namespace std;

#pragma mark Composition functions

double compositionFunctionMain::sum(vector<NeuralConnection*> *x) {
    double sum = 0;
    for (unsigned long i = 0; i < x->size(); i++) {
        sum += x->at(i)->weightedValue();
    }
    return sum;
}

double compositionFunctionMain::dist(vector<NeuralConnection*> *x) {
    return 0;
}


#pragma mark - Composition function derivatives

double compositionFunctionDerivative::sum(double *x) {
    return 1;
}

double compositionFunctionDerivative::dist(double *x) {
    return 1;
}

#pragma mark - Activation functions

double activationFunctionMain::sigmoid(double *x) {
    return 1.0 / (1.0 + exp(-*x));
}

double activationFunctionMain::heavyside(double *x) {
    if (x < 0)
        return 0;
    else
        return 1;
}

double activationFunctionMain::tanh(double *x) {
    return ::tanh(*x);
}

double activationFunctionMain::linear(double *x) {
    return *x;
}

double activationFunctionMain::relu(double *x) {
    if (*x < 0)
        return 0;
    return *x;
}

#pragma mark - Activation function derivatives

double activationFunctionDerivative::sigmoid(double *x) {
    return *x * (1 - *x);
}

double activationFunctionDerivative::heavyside(double *x) {
    if (abs(*x) == 0.0) {
        return HUGE_VAL;
    }
    return 0;
}

double activationFunctionDerivative::tanh(double *x) {
    return 1 - *x * *x;
}

double activationFunctionDerivative::linear(double *x) {
    return 1;
}

double activationFunctionDerivative::relu(double *x) {
    if (*x < 0)
        return 0;
    return 1;
}

#pragma mark - Cost functions

double costFunctionMain::quadratic(std::vector<double> *x1, std::vector<double> *x2) {
    double value = 0;
    for (UInteger i = 0; i < x1->size(); i++)
        value += (x1->at(i) - x2->at(i)) * (x1->at(i) - x2->at(i));
    return sqrt(value);
}

#pragma mark - Cost function derivatives

double costFunctionDerivative::quadratic(double *x1, double *x2) {
    return *x1 - *x2;
}



#pragma mark - Function wrappers

CompositionFunction makeCompositionFunction(CompositionFunctionMain main, CompositionFunctionDerivative derivative) {
    CompositionFunction composition;
    composition.id = CompositionFunctionTypeOther;
    composition.main = main;
    composition.derivative = derivative;
    return composition;
}

const CompositionFunction compositionFunctionSum = {CompositionFunctionTypeSum, &compositionFunctionMain::sum, &compositionFunctionDerivative::sum};

const CompositionFunction compositionFunctionDist {CompositionFunctionTypeDist, &compositionFunctionMain::dist, &compositionFunctionDerivative::dist};


ActivationFunction makeActivationFunction(ActivationFunctionMain main, ActivationFunctionDerivative derivative) {
    ActivationFunction activation;
    activation.id = ActivationFunctionTypeOther;
    activation.main = main;
    activation.derivative = derivative;
    return activation;
}

const ActivationFunction activationFunctionHeavyside{ActivationFunctionTypeHeavyside, &activationFunctionMain::heavyside, &activationFunctionDerivative::heavyside};

const ActivationFunction activationFunctionSigmoid {ActivationFunctionTypeSigmoid, &activationFunctionMain::sigmoid, &activationFunctionDerivative::sigmoid};

const ActivationFunction activationFunctionTanh {ActivationFunctionTypeArctan, &activationFunctionMain::tanh, &activationFunctionDerivative::tanh};

const ActivationFunction activationFunctionLinear {ActivationFunctionTypeLinear, &activationFunctionMain::linear, &activationFunctionDerivative::linear};

const ActivationFunction activationFunctionReLu {ActivationFunctionTypeReLu, &activationFunctionMain::relu, &activationFunctionDerivative::relu};


CostFunction makeCostFunction(CostFunctionMain main, CostFunctionDerivative derivative) {
    CostFunction cost;
    cost.id = CostFunctionTypeOther;
    cost.main = main;
    cost.derivative = derivative;
    return cost;
}

const CostFunction costFunctionQuadratic {CostFunctionTypeQuadratic, &costFunctionMain::quadratic, &costFunctionDerivative::quadratic};
