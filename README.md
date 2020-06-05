# neuralnetwork

This projects is a dummy neural network example. Using the MNIST database a neural net is trained that is capable of recognizing hand written digits. In this example the neural network is trained to distinguish zero from non zero digits.

The frontend of the application is a Windows Presentation Foundation app, and the backend is encapsulated in a dll written in C++. This app can be used only for training, with no feedback.

The backend_app is a simple C++ console application that is capable of training and testing the neural net. This also using the backend dll.

In order to try it, please do the following steps:
* Download the MNIST database from here and extract it: http://yann.lecun.com/exdb/mnist/.
* After extracting specify the folder that contains the database in backend/backend.h by setting MNIST_path variable to the proper value.
* Open backend/backend.sln and set the solution platforms to x86
* Build the solution, this generates the backend.dll to the output directory
* Then open backend_app/backend_app.sln and frontend_app/frontend_app.sln and build them (in case of the backend_app also on x86 platform!)
* Copy the dll into the output directory of each solution.
* Run any of the two application. In case of backend app to change between train and test mode define or undefine TRAIN macro at the top of backend.h.
