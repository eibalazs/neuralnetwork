#pragma once

#include <vector>
#include "matrix.h"

typedef std::vector<double> Image;
typedef Matrix<double> MNISTimages;

typedef double Label;
typedef std::vector<Label> MNISTlabels;

typedef std::vector<double> Weights;