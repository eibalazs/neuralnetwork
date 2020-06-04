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

namespace EIB_GE_test
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            //int c = add(5, 43);
            //Console.WriteLine(c);
        }

        private void LoadDataClick(object sender, RoutedEventArgs e)
        {
            textBox.Text = "LoadDataClick";
            loadDataButton.Content = "Clicked";
            Task.Factory.StartNew(() =>
            {
                Dispatcher.Invoke(() =>
                {
                    trainButton.Content = add(3, 63).ToString();
                });
            });
        }

        private void TrainClick(object sender, RoutedEventArgs e)
        {

        }
        private void PauseClick(object sender, RoutedEventArgs e)
        {

        }

        [DllImport("AI_library.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void readMNISTdata(string path);

        [DllImport("AI_library.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int add(int a, int b);
    }
}
