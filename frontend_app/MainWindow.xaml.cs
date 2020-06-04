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
        bool train_runs = false;
        bool train_paused = false;
        Thread training_thread;
        public MainWindow()
        {
            InitializeComponent();
            trainButton.IsEnabled = false;
            pauseButton.IsEnabled = false;

            Bitmap bitmap = new Bitmap(28,28);
            var color = System.Drawing.Color.Green;

            for (int i = 0; i<28; ++i)
            {
                for(int j = 0; j < 28;++j)
                    bitmap.SetPixel(i, j, color);

            }

            image.Source = BitmapToImageSource(bitmap);
        }

        private BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
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
            train_runs = !train_runs;

            if (train_runs)
            {
                loadDataButton.IsEnabled = false;
                trainButton.Content = "Stop training";
                pauseButton.IsEnabled = true;

                initializeTraining();

                training_thread = new Thread(trainNeuralNet);
                training_thread.Start();
            }
            else
            {
                loadDataButton.IsEnabled = true;
                trainButton.Content = "Train neural net";
                pauseButton.IsEnabled = false;

                training_thread.Abort();
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
