#include "pch.h"
#include <algorithm>
#include <fstream>
#include "backend.h"


int main()
{
    printf("Welcome to the neural network trainer/tester program!\n");
    printf("To change between train and test mode define or undefine TRAIN macro at the top of backend.h!\n");
    printf("Loading data...\n");

    loadMNISTimages();
    printf("X readed successfully.\n");

    loadMNISTlabels();
    printf("Y readed successfully.\n");

    /* This is the number of pixels within an image */
    n_x = X.getNumberOfColumns();
    /* This is the number of images in the data set */
    m = X.getNumberOfRows();

#ifdef TRAIN

    printf("The current mode is training.\n");
    initializeTraining();
    printf("Training initialized.\n");
    printf("Start training...\n");
    trainNeuralNet();

#else

    printf("The current mode is testing.\n");

    importWeightsFromCSV();
    importBiasFromCSV();

    testNeuralNet();

#endif


    //const auto X_test = loadMNISTimages(root_folder + "/t10k-images.idx3-ubyte");
    //const auto Y_test = loadMNISTlabels(root_folder + "/t10k-labels.idx1-ubyte");
}

double getCost() { return cost; };

void reverseInt(int& i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    i = ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void loadMNISTimages() 
{
#ifdef TRAIN
    std::ifstream file(MNIST_path + "/train-images.idx3-ubyte", std::ios::binary);
#else
    std::ifstream file(MNIST_path + "/t10k-images.idx3-ubyte", std::ios::binary);
#endif

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

        X = mnist_images;
    }
    else throw std::runtime_error("Could not open MNIST image file at " + MNIST_path);
}

void loadMNISTlabels()
{
#ifdef TRAIN
    std::ifstream file(MNIST_path + "/train-labels.idx1-ubyte", std::ios::binary);
#else
    std::ifstream file(MNIST_path + "/t10k-labels.idx1-ubyte", std::ios::binary);
#endif

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

        Y = mnist_labels;
    }
    else throw std::runtime_error("Could not open MNIST label file at " + MNIST_path);
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

void initializeTraining()
{
    learning_rate = 1.0;

    std::random_device random_device;
    std::mt19937 mersenne_engine{ random_device() };
    std::normal_distribution<double> distribution{ 0.0, 1.0 };

    auto generate_random = [&]() {
        return 0.01 * distribution(mersenne_engine);
    };

    W = Weights(n_x);
    std::generate(std::begin(W), std::end(W), generate_random);

    dW = Weights(n_x);

    b = 0.0;
    db = 0.0;
}

void trainNeuralNet()
{
    while (true) 
    {
        std::vector<double> Z(m);
        std::vector<double> A(m);

        for (size_t j = 0; j < m; ++j) {
            Z[j] = W * X[j] + b;
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

        printf("Cost: %f\n", cost);
        exportWeightsToCSV();
        exportBiasToCSV();
    }
}

void testNeuralNet()
{
    std::vector<double> Z(m);
    std::vector<double> A(m);

    for (size_t j = 0; j < m; ++j) {
        Z[j] = W * X[j];
    }

    std::transform(std::begin(Z), std::end(Z), std::begin(A), [](const double& z) {
        return sigmoid(z); });

    unsigned int num_of_0_correct = 0;
    unsigned int num_of_non_0_correct = 0;
    unsigned int num_of_0_incorrect = 0;
    unsigned int num_of_non_0_incorrect = 0;

    for (size_t k = 0; k < m; ++k) {
        if (Y[k] == 0 && A[k] <= 0.5)
            num_of_non_0_correct++;
        else if (Y[k] == 0 && A[k] > 0.5)
            num_of_non_0_incorrect++;
        else if (Y[k] == 1 && A[k] <= 0.5)
            num_of_0_incorrect++;
        else num_of_0_correct++;
    }

    printf("Confusion matrix:\n%i  %i\n%i  %i\n", num_of_non_0_correct, num_of_0_incorrect, num_of_non_0_incorrect, num_of_0_correct);
    system("pause");
}

void exportWeightsToCSV()
{
    std::ofstream csv;
    csv.open("W.csv");
    std::copy(std::begin(W), std::end(W), std::ostream_iterator<double>(csv, "\n"));
    csv.close();
}

void exportBiasToCSV()
{
    std::ofstream csv;
    csv.open("b.csv");
    csv << b;
    csv.close();
}

void importWeightsFromCSV()
{
    std::ifstream csv;
    csv.open("W.csv");
    if (csv.is_open()) {
        W.clear();
        printf("Weight csv is opened.\n");
        std::string str;
        while (std::getline(csv, str))
        {
            W.push_back(stod(str));
        }
    }
    else printf("Weight csv not found!\n");
}

void importBiasFromCSV()
{
    std::ifstream csv;
    csv.open("b.csv");
    if (csv.is_open()) {
        printf("Bias csv is opened.\n");
        std::string str;
        std::getline(csv, str);
        b = stod(str);
    }
    else printf("Bias csv not found!\n");
}
