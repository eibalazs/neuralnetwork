using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Threading;
using System.Drawing;
using System.IO;

namespace EIB_GE_test
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    /// 

    public partial class MainWindow : Window
    {
        bool train_clicked = false;
        bool train_paused = false;
        bool train_runs = false;
        Thread training_thread;

        public MainWindow()
        {
            InitializeComponent();
            trainButton.IsEnabled = false;
            pauseButton.IsEnabled = false;
        }

        private void LoadDataClick(object sender, RoutedEventArgs e)
        {
            loadDataButton.Content = "Loading data...";
            loadDataButton.IsEnabled = false;
            trainButton.IsEnabled = false;
            pauseButton.IsEnabled = false;
            Task.Factory.StartNew(() =>
            {
                loadMNISTlabels();
                loadMNISTimages();

                Dispatcher.Invoke(() =>
                {
                    loadDataButton.Content = "Load data set";
                    loadDataButton.IsEnabled = true;
                    trainButton.IsEnabled = true;
                    pauseButton.IsEnabled = false;
                });
            });
        }

        private void TrainClick(object sender, RoutedEventArgs e)
        {
            train_clicked = !train_clicked;

            if (train_clicked)
            {
                loadDataButton.IsEnabled = false;
                trainButton.Content = "Stop training";
                pauseButton.IsEnabled = true;

                initializeTraining();

                training_thread = new Thread(trainNeuralNet);
                training_thread.Start();
                train_runs = true;
            }
            else
            {
                loadDataButton.IsEnabled = true;
                trainButton.Content = "Train neural net";
                pauseButton.IsEnabled = false;

                training_thread.Abort();
                train_runs = false;
            }
        }

        private void PauseClick(object sender, RoutedEventArgs e)
        {
            train_paused = !train_paused;

            if (train_paused)
            {
                pauseButton.Content = "Resume";
                training_thread.Suspend();
            }
            else
            {
                pauseButton.Content = "Pause";
                training_thread.Resume();
            }
        }

        [DllImport("backend.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void loadMNISTimages();

        [DllImport("backend.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void loadMNISTlabels();

        [DllImport("backend.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void initializeTraining();

        [DllImport("backend.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void trainNeuralNet();
    }
}
