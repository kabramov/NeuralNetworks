using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;


namespace xorNet
{

    public interface INeuron
    {
        double Result { get; set; }
        double Delta { get; set; }
        Dictionary<INeuron, double> Weights { get; }
        void Run(INeuralLayer layer);
    }

    public interface INeuralLayer
    {
        List<INeuron> Neurons { get; }
        void Run(INeuralNetwork net);
    }

    public interface INeuralNetwork
    {
        INeuralLayer InputLayer { get; }
        INeuralLayer HiddenLayer { get; }
        INeuralLayer OutputLayer { get; }
        void Run();
    }

    public class Neuron : INeuron
    {
        public Neuron(double x)
        {
            result = x;
            weights = new Dictionary<INeuron, double>();
        }
        public Neuron()
        {
            weights = new Dictionary<INeuron, double>();
        }

        public double result;
        public double delta;
        public Dictionary<INeuron, double> weights;

        public Dictionary<INeuron, double> Weights
        {
            get { return weights; }
        }

        public double Delta
        {
            get { return delta; }
            set { delta = value; }
        }

        public double Result
        {
            get { return result; }
            set { result = value; }
        }

        public void Run(INeuralLayer layer)
        {
            result = 0;
            foreach (var x in weights)
            {
                result += x.Key.Result * x.Value;
            }
            result = Sigmoid(result);
        }

        private double Sigmoid(double net)
        {
            return 1 / (1 + Math.Exp(-net));
        }
    }

    public class NeuralLayer : INeuralLayer
    {
        public List<INeuron> neurons;
        public NeuralLayer()
        {
            neurons = new List<INeuron>();
        }
        public void Run(INeuralNetwork Net)
        {
            foreach (var n in neurons)
                n.Run(this);
        }

        public List<INeuron> Neurons
        {
            get { return neurons; }
        }
    }

    public class NeuralNetwork : INeuralNetwork
    {
        public INeuralLayer inputlayer;
        public INeuralLayer hiddenlayer;
        public INeuralLayer outputlayer;

        public INeuralLayer InputLayer
        {
            get { return inputlayer; }
        }
        public INeuralLayer HiddenLayer
        {
            get { return hiddenlayer; }
        }
        public INeuralLayer OutputLayer
        {
            get { return outputlayer; }
        }
        public void Run()
        {
            hiddenlayer.Run(this);
            outputlayer.Run(this);
        }

        public void initialize(double x1, double x2)
        {
            start(this, x1, x2);
        }

        public void initialize( int N1, int N2)
        {
            start(this, N1, N2);
        }

        private void start(NeuralNetwork net, double x1, double x2)
        {
            net.inputlayer = new NeuralLayer();
            net.hiddenlayer = new NeuralLayer();
            net.outputlayer = new NeuralLayer();

            net.inputlayer.Neurons.Add(new Neuron(x1));
            net.inputlayer.Neurons.Add(new Neuron(x2));
            net.inputlayer.Neurons.Add(new Neuron(1));

            net.hiddenlayer.Neurons.Add(new Neuron());
            net.hiddenlayer.Neurons.Add(new Neuron());
            net.hiddenlayer.Neurons.Add(new Neuron(1));
            Random rand = new Random(DateTime.Now.Millisecond);
            net.hiddenlayer.Neurons[0].Weights.Add(net.inputlayer.Neurons[0], nextWeight(rand));
            net.hiddenlayer.Neurons[0].Weights.Add(net.inputlayer.Neurons[1], nextWeight(rand));
            net.hiddenlayer.Neurons[0].Weights.Add(net.inputlayer.Neurons[2], nextWeight(rand));
            net.hiddenlayer.Neurons[1].Weights.Add(net.inputlayer.Neurons[0], nextWeight(rand));
            net.hiddenlayer.Neurons[1].Weights.Add(net.inputlayer.Neurons[1], nextWeight(rand));
            net.hiddenlayer.Neurons[1].Weights.Add(net.inputlayer.Neurons[2], nextWeight(rand));

            net.outputlayer.Neurons.Add(new Neuron());

            net.outputlayer.Neurons[0].Weights.Add(net.hiddenlayer.Neurons[0], nextWeight(rand));
            net.outputlayer.Neurons[0].Weights.Add(net.hiddenlayer.Neurons[1], nextWeight(rand));
            net.outputlayer.Neurons[0].Weights.Add(net.hiddenlayer.Neurons[2], nextWeight(rand));
        }

        private void start(NeuralNetwork net, int N1, int N2)
        {
            net.inputlayer = new NeuralLayer();
            net.hiddenlayer = new NeuralLayer();
            net.outputlayer = new NeuralLayer();

            for (int i = 0; i < N1; i++)
            {
                net.inputlayer.Neurons.Add(new Neuron());
            }

            for (int i = 0; i < N1; i++)
            {
                net.hiddenlayer.Neurons.Add(new Neuron());
            }
    
            Random rand = new Random();

            for (int i = 0; i < N1; i++)
            {
                for (int j = 0; j < N1; j++)
                {
                    net.hiddenlayer.Neurons[j].Weights.Add(net.inputlayer.Neurons[i], net.nextWeight(rand));
                }
            }

            for (int i = 0; i < N2; i++)
            {
                net.outputlayer.Neurons.Add(new Neuron());
            }

            for (int i = 0; i < N1; i++)
            {
                for (int j = 0; j < N2; j++)
                {
                    net.outputlayer.Neurons[j].Weights.Add(net.hiddenlayer.Neurons[i], net.nextWeight(rand));
                }
            }
        }

        private double nextWeight(Random rand)
        {
            return rand.NextDouble() * 0.6 - 0.3;
        }

        public void train(List <int> t, double n)
        {
            training(this, t, n);
        }

        private void training(NeuralNetwork net, List<int> t, double n)
        {
            double o;
            for (int j = 0; j < t.Count; j++)
            {
                o = net.outputlayer.Neurons[j].Result;
                net.outputlayer.Neurons[j].Delta = (t[j] - o) * o * (1 - o);
            }
            var keys = new List<INeuron>();
            for (int j = 0; j < t.Count; j++)
            {
                keys = new List<INeuron>(net.outputlayer.Neurons[j].Weights.Keys);
                foreach (var key in keys)
                {
                    net.outputlayer.Neurons[j].Weights[key] += n * key.Result * net.outputlayer.Neurons[j].Delta;
                }
            }
            keys = new List<INeuron>(net.outputlayer.Neurons[0].Weights.Keys);
            foreach (var key in keys)
            {
                double sum = 0;
                for (int j = 0; j < t.Count; j++)
                {
                    sum += net.outputlayer.Neurons[j].Delta * net.outputlayer.Neurons[j].Weights[key];
                }
                key.Delta = key.Result * (1 - key.Result) * sum;
            }
                       
            for (int j = 0; j < t.Count; j++)
            {
                keys = new List<INeuron>(net.hiddenlayer.Neurons[j].Weights.Keys);
                foreach (var key in keys)
                {
                    net.hiddenlayer.Neurons[j].Weights[key] += n * key.Result * net.hiddenlayer.Neurons[j].Delta;
                }
            }
        }

        public bool checkResult(double t)
        {
            Program pr = new Program();
            double sum = 0;
            string s = pr.DoubleToStr(t, 4, 0);
            for (int i = 1; i < s.Length - 1; i++)
            {
                if (s[i] == '0')
                    sum += Math.Pow(this.outputlayer.Neurons[i-1].Result, 2.0);
                else sum += Math.Pow(1-this.outputlayer.Neurons[i-1].Result, 2.0);
            }
            return sum/4 < 0.001;
        }

        public List<double> GetWeights()
        {
            List<double> l = new List<double>();
            foreach (var x in this.hiddenlayer.Neurons[0].Weights)
            {
                l.Add(x.Value);
            }
            foreach (var x in this.hiddenlayer.Neurons[1].Weights)
            {
                l.Add(x.Value);
            }
            return l;
        }
    }

    class Program
    {
        public string DoubleToStr(double x, int n1, int n2)
        {
            string s = "";
            if (x < 0)
            {
                s += "-";
            }
            else
            {
                s += "+";
            }
            s += IntToBit(Math.Abs(x), n1) + ".";
            s += FracToBit(Math.Abs(x), n2);
            return s;
        }

        public string IntToBit(double x, int n)
        {
            double number = Math.Truncate(x);
            string s = "";
            while (number > 0)
            {
                s = String.Concat(Convert.ToString(number % 2), s);
                number = Math.Truncate(number / 2);
            }
            while (s.Length < n)
            {
                s = String.Concat("0", s);
            }
            return s;
        }

        public string FracToBit(double x, int len)
        {
            double number = x - Math.Truncate(x);
            string str = "";
            int t;
            int i = 0;
            while (i < len)
            {
                number *= 2;
                t = Convert.ToInt32(Math.Truncate(number));
                str = String.Concat(str, Convert.ToString(t));
                number -= t;
                i++;
            }
            return str;
        }

        public double StrToDouble(string s)
        {
            double res = 0;
            string[] a = s.Split('.');
            int n = a[0].Length;
            if (a[0].Contains('+'))
            {
                for (int i = 1; i < a[0].Length; ++i)
                {
                    res += Convert.ToDouble(a[0][i].ToString()) * Math.Pow(2, n - i - 1);
                }

                if (a.Length > 1)
                {
                    for (int i = 0; i < a[1].Length; ++i)
                    {
                        res += Convert.ToDouble(a[1][i].ToString()) * Math.Pow(2, -(i + 1));
                    }
                }
            }
            else
            {
                for (int i = 1; i < a[0].Length; ++i)
                {
                    res -= Convert.ToDouble(a[0][i].ToString()) * Math.Pow(2, n - i - 1);
                }

                if (a.Length > 1)
                {
                    for (int i = 0; i < a[1].Length; ++i)
                    {
                        res -= Convert.ToDouble(a[1][i].ToString()) * Math.Pow(2, -(i + 1));
                    }
                }
            }
            return res;
        }

        static void Main(string[] args)
        {
            Program pr = new Program();
            Dictionary<string, double> Data = new Dictionary<string, double>();
            for (int i = 0; i < 10; i++)
            {
                Data.Add(i.ToString()+".jpg", i);
            }
            int N = 20;//размер изображения
            Random rand = new Random();
            NeuralNetwork xornet = new NeuralNetwork();
            xornet.initialize(N*N,4);
            int iterations = 0;
            List<int> numbers = new List<int> {0,0,0,0};
            bool stop = false;
            while (!stop)
            {
                stop = true;
                foreach (var x in Data)
                {
                    Bitmap bmp = new Bitmap(@"D:\numbers\"+ x.Key);
                    for (int i = 0; i < N; i++)
                    {
                        for (int j = 0; j < N; j++)
                        {
                            if (bmp.GetPixel(i,j).Name =="ff000000")
                                xornet.inputlayer.Neurons[i * N + j].Result = 1;
                            else xornet.inputlayer.Neurons[i * N + j].Result = 0;
                        }
                    }
                    xornet.Run();
                    if (xornet.checkResult(x.Value))
                    {
                        continue;
                    }
                    stop = false;
                    string s = pr.DoubleToStr(x.Value, 4, 0);
                    for (int i=1;i<s.Length-1;i++)
                    {
                        if (s[i] == '0')
                            numbers[i-1] = 0;
                        else numbers[i-1] = 1;
                    }
                    xornet.train(numbers, 0.3);
                    Console.WriteLine(iterations);
                    ++iterations;
                }         
            }
            Console.WriteLine("Training completed: {0} iterations", iterations);
            foreach (var x in Data)
            {
                Bitmap bmp = new Bitmap(@"D:\numbers\"+x.Key);
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        if (bmp.GetPixel(i, j).Name == "ff000000")
                            xornet.inputlayer.Neurons[i * N + j].Result = 1;
                        else xornet.inputlayer.Neurons[i * N + j].Result = 0;
                    }
                }
                xornet.Run();
                string s = "+";
                foreach (var t in xornet.outputlayer.Neurons)
                {
                    if (t.Result >= 0.95)
                        s += '1';
                    else s += '0';
                }
                double result = pr.StrToDouble(s);
                Console.WriteLine("image={0}, net={1} ", x.Value, result);
            }

        }
    }
}
