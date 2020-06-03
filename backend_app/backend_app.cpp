// backend_app.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "AI_library.h"

int main()
{
    const std::string root_folder = "C:/work/neuralnetwork/data/";

    const auto X_train = loadMNISTimages(root_folder + "/train-images.idx3-ubyte");
    printf("X_train readed successfully.\n");

    const auto Y_train = loadMNISTlabels(root_folder + "/train-labels.idx1-ubyte");
    printf("Y_train readed successfully.\n");

    const auto weights = trainNeuralNet(X_train, Y_train);

    exportWeightsToCSV(weights);

    //const auto X_test = loadMNISTimages(root_folder + "/t10k-images.idx3-ubyte");
    //const auto Y_test = loadMNISTlabels(root_folder + "/t10k-labels.idx1-ubyte");
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
