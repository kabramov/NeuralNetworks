using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
            //return (net >= 0) ? 1 : 0;
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
            Random rand = new Random();           
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

        private double nextWeight(Random rand)
        {
            return rand.NextDouble() * 0.6 - 0.3;
        }

        public void train(double t, double n)
        {
            training(this, t, n);
        }

        private void training(NeuralNetwork net, double t, double n)
        {
            double o = net.outputlayer.Neurons[0].Result;
            net.outputlayer.Neurons[0].Delta = (t - o)*o*(1-o);
            int i = 0;
            var keys = new List<INeuron>(net.outputlayer.Neurons[0].Weights.Keys);
            foreach (var key in keys)
            {
                net.outputlayer.Neurons[0].Weights[key] += n * key.Result * net.outputlayer.Neurons[0].Delta;
                net.hiddenlayer.Neurons[i].Delta = net.hiddenlayer.Neurons[i].Result * (1 - net.hiddenlayer.Neurons[i].Result) * net.outputlayer.Neurons[0].Delta * net.outputlayer.Neurons[0].Weights[key];
                ++i;
            }
            keys = new List<INeuron>(net.hiddenlayer.Neurons[0].Weights.Keys);
            foreach (var key in keys)
            {
                net.hiddenlayer.Neurons[0].Weights[key] += n * key.Result * net.hiddenlayer.Neurons[0].Delta;
            }
            keys = new List<INeuron>(net.hiddenlayer.Neurons[1].Weights.Keys);
            foreach (var key in keys)
            {
                net.hiddenlayer.Neurons[1].Weights[key] += n * key.Result * net.hiddenlayer.Neurons[1].Delta;
            }
        }

        public bool checkResult(double t)
        {
            return Math.Abs(t - this.outputlayer.Neurons[0].Result) < 0.1;
        }         
    }  

    class Program
    {     
        static void Main(string[] args)
        {
            Dictionary<double[], double> Data = new Dictionary<double[], double>(); 
            Data.Add(new double[] { 0, 0 },0);
            Data.Add(new double[] { 0, 1 }, 1);
            Data.Add(new double[] { 1, 0 }, 1);
            Data.Add(new double[] { 1, 1 }, 0);
            Random rand = new Random();
            NeuralNetwork xornet = new NeuralNetwork();
            int iterations = 0;
            bool stop = false;
            xornet.initialize(0, 0);
            while (!stop)
            {
                stop = true;
                foreach (var x in Data)       
                {
                    xornet.inputlayer.Neurons[0].Result = x.Key[0];
                    xornet.inputlayer.Neurons[1].Result = x.Key[1];
                    xornet.Run();
                    if (xornet.checkResult(x.Value))
                    {
                        continue;
                    } 
                    stop = false;
                    xornet.train(x.Value, 0.3);
                    ++iterations;
                }
            }
            Console.WriteLine("Training completed: {0} iterations", iterations);
            foreach (var x in Data)
            {
                xornet.inputlayer.Neurons[0].Result = x.Key[0];
                xornet.inputlayer.Neurons[1].Result = x.Key[1];
                xornet.Run();
                Console.WriteLine("{0} xor {1} = {2} ({3})", x.Key[0], x.Key[1], xornet.OutputLayer.Neurons[0].Result, x.Value);
            }           
        }
    }
}
