#pragma once

#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>

#include "types.h"

static std::string MNIST_path = "C:/work/neuralnetwork/data";

static MNISTimages X_train;

static MNISTlabels Y_train;

static Weights weights;

static Weights dW;

static double learning_rate;

static size_t n_x;
static size_t m;

static double b;
static double db;

double cost;

#ifdef BACKEND_EXPORTS
#define BACKEND_API __declspec(dllexport)
#else
#define BACKEND_API __declspec(dllimport)
#endif // BACKEND_EXPORTS

/**
 * Backend API
 */
	
extern "C" BACKEND_API int main();

extern "C" BACKEND_API void loadMNISTimages();

extern "C" BACKEND_API void loadMNISTlabels();

extern "C" BACKEND_API void initializeTraining();

extern "C" BACKEND_API void trainNeuralNet();

inline MNISTlabels operator-(const MNISTlabels& lhs, const MNISTlabels& rhs) {
	if (lhs.size() != rhs.size())
		throw std::runtime_error("Cannot subtract vectors with different sizes!");
	MNISTlabels result(lhs.size());
	for (size_t i = 0; i < rhs.size(); ++i) {
		result[i] = lhs[i] - rhs[i];
	}
	return result;
}

template<typename T>
inline std::vector<double> operator*(const double& lhs, const std::vector<T>& rhs) {
	MNISTlabels result(rhs.size());
	for (size_t i = 0; i < rhs.size(); ++i) {
		result[i] = lhs * rhs[i];
	}
	return result;
}

void reverseInt(int& i);

inline double sigmoid(const double& x) { return 1.0 / (1.0 + exp(-x)); }

template<typename T, typename U> 
double operator*(const std::vector<T>& a, const std::vector<U>& b)
{
	if (a.size() != b.size()) {
		throw std::runtime_error("The size of the vectors given as parameters for dot product are not equal!");
	}
	return std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.0);
}

double computeLoss(const std::vector<double>& y, const std::vector<double>& y_hat);

void exportWeightsToCSV(const Weights& weights);