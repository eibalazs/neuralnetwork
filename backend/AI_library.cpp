#include "pch.h"
#include <algorithm>
#include <fstream>
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

void reverseInt(int& i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    i = ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

MNISTimages loadMNISTimages(const std::string& path) 
{
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) 
    {
        int magic_number, n_images, n_rows, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        reverseInt(magic_number);
        file.read((char*)&n_images, sizeof(n_images));
        reverseInt(n_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        reverseInt(n_cols);

        auto mnist_images = MNISTimages(n_images, n_rows * n_cols);

        for (int i = 0; i < n_images; ++i) {
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char pixel = 0;
                    file.read((char*)&pixel, sizeof(pixel));
                    mnist_images.fillData(i, r * n_cols + c, static_cast<double>(pixel) / 255.0);
                }
            }
        }

        return mnist_images;
    }
    else throw std::runtime_error("Could not open MNIST image file at " + path);
}

MNISTlabels loadMNISTlabels(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) 
    {
        int magic_number, n_labels = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        reverseInt(magic_number);
        file.read((char*)&n_labels, sizeof(n_labels));
        reverseInt(n_labels);

        auto mnist_labels = MNISTlabels(n_labels, Label());

        for (int i = 0; i < n_labels; ++i)
        {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            mnist_labels[i] = label == 0 ? 1 : 0;
        }

        return mnist_labels;
    }
    else throw std::runtime_error("Could not open MNIST label file at " + path);
}

double computeLoss(const std::vector<double>& Y, const std::vector<double>& Y_hat) 
{
    if (Y.size() != Y_hat.size()) {
        throw std::runtime_error("The size of the two vector used for loss calculation is not equal!");
    }

    const auto size = Y.size();

    auto loss_vector = std::vector<double>(size);

    std::transform(Y.begin(), Y.end(), Y_hat.begin(), loss_vector.begin(), [](const double& y, const double& y_hat) {
        return y * log(y_hat) + (1 - y)*log(1 - y_hat); });

    const auto loss = -(1.0 / size) * std::accumulate(loss_vector.begin(), loss_vector.end(), 0.0);

    return loss;
}

Weights trainNeuralNet(const MNISTimages& X, const MNISTlabels& Y)
{
    const double learning_rate = 1.0;
    const size_t num_of_iterations = 2000;
    /* This is the number of pixels within an image */
    const auto n_x = X.getNumberOfColumns();
    /* This is the number of images in the data set */
    const auto m = X.getNumberOfRows();

    std::random_device random_device;
    std::mt19937 mersenne_engine{ random_device() };
    std::normal_distribution<double> distribution{ 0.0, 1.0 };

    auto generate_random = [&]() {
        return 0.01 * distribution(mersenne_engine);
    };

    auto W = Weights(n_x);
    auto dW = Weights(n_x);

    std::generate(std::begin(W), std::end(W), generate_random);
    double b = 0.0;
    double db = 0.0;

    double cost;

    for (int i = 0; i < num_of_iterations; ++i) 
    {
        std::vector<double> Z(m);
        std::vector<double> A(m);

        for (size_t j = 0; j < m; ++j) {
            Z[j] = W * X[j];
        }

        std::transform(std::begin(Z), std::end(Z), std::begin(A), [](const double& z) {
            return sigmoid(z); });

        cost = computeLoss(Y, A);

        const auto diff = A - Y;

        for (size_t k = 0; k < n_x; ++k) {
            dW[k] = (1.0 / m) * X.getColumn(k) * (diff);
        }

        db = (1.0 / m) * std::accumulate(std::begin(diff), std::end(diff), 0.0);

        W = W - learning_rate * dW;
        b = b - learning_rate * db;

        printf("Epoch %i, cost %f\n", i, cost);
    }

    printf("Final cost: %f\n", cost);

    return W;
}

void exportWeightsToCSV(const Weights& weights)
{
    std::ofstream csv;
    csv.open("W.csv");
    std::copy(std::begin(weights), std::end(weights), std::ostream_iterator<double>(csv, "\n"));
    csv.close();
}
