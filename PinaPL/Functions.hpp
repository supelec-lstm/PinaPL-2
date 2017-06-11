//
//  mathFunctions.hpp
//  PinaPL
//

#ifndef Functions_hpp
#define Functions_hpp

#include <stdio.h>
#include <vector>

#include "PinaPLDefines.hpp"


class NeuralConnection;

typedef double (*CompositionFunctionMain) (std::vector<NeuralConnection*>*);
typedef double (*CompositionFunctionDerivative) (double*);
typedef double (*ActivationFunctionMain) (double*);
typedef double (*ActivationFunctionDerivative) (double*);
typedef double (*CostFunctionMain) (std::vector<double>*, std::vector<double>*);
typedef double (*CostFunctionDerivative) (double*, double*);


enum CompositionFunctionType : UInteger {
    CompositionFunctionTypeSum = 1,
    CompositionFunctionTypeDist = 2,
    CompositionFunctionTypeOther = NotFound
};

enum ActivationFunctionType : UInteger {
    ActivationFunctionTypeHeavyside = 1,
    ActivationFunctionTypeSigmoid = 2,
    ActivationFunctionTypeTanh = 3,
    ActivationFunctionTypeLinear = 4,
    ActivationFunctionTypeReLu = 5,
    ActivationFunctionTypeOther = NotFound
};

enum CostFunctionType : UInteger {
    CostFunctionTypeQuadratic = 1,
    CostFunctionTypeOther = NotFound
};


struct CompositionFunction {
    CompositionFunctionType id;
    CompositionFunctionMain main;
    CompositionFunctionDerivative derivative;
};

struct ActivationFunction {
    ActivationFunctionType id;
    ActivationFunctionMain main;
    ActivationFunctionDerivative derivative;
};

struct CostFunction {
    CostFunctionType id;
    CostFunctionMain main;
    CostFunctionDerivative derivative;
};


namespace compositionFunctionMain {
    double sum(std::vector<NeuralConnection*> *x);
    double dist(std::vector<NeuralConnection*> *x);
}

namespace compositionFunctionDerivative {
    double sum(double *x);
    double dist(double *x);
}

namespace activationFunctionMain {
    double heavyside(double *x);
    double sigmoid(double *x);
    double tanh(double *x);
    double linear(double *x);
    double relu(double *x);
}

namespace activationFunctionDerivative {
    double heavyside(double *x);
    double sigmoid(double *x);
    double tanh(double *x);
    double linear(double *x);
    double relu(double *x);
}

namespace costFunctionMain {
    double quadratic(std::vector<double> *x1, std::vector<double> *x2);
}

namespace costFunctionDerivative {
    double quadratic(double *x1, double *x2);
}


CompositionFunction makeCompositionFunction(CompositionFunctionMain main, CompositionFunctionDerivative derivative);
extern const CompositionFunction compositionFunctionSum;
extern const CompositionFunction compositionFunctionDist;

ActivationFunction makeActivationFunction(ActivationFunctionMain main, ActivationFunctionDerivative derivative);
extern const ActivationFunction activationFunctionHeavyside;
extern const ActivationFunction activationFunctionSigmoid;
extern const ActivationFunction activationFunctionTanh;
extern const ActivationFunction activationFunctionLinear;
extern const ActivationFunction activationFunctionReLu;

CostFunction makeCostFunction(CostFunctionMain main, CostFunctionDerivative derivative);
extern const CostFunction costFunctionQuadratic;

#endif /* Functions_hpp */
