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

        public void initialize(double x1, double x2,List<double> l)
        {
            start(this, x1, x2,l);
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

        private void start(NeuralNetwork net, double x1, double x2, List<double> l)
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
            net.hiddenlayer.Neurons[0].Weights.Add(net.inputlayer.Neurons[0], l[0]);
            net.hiddenlayer.Neurons[0].Weights.Add(net.inputlayer.Neurons[1], l[1]);
            net.hiddenlayer.Neurons[0].Weights.Add(net.inputlayer.Neurons[2], l[2]);
            net.hiddenlayer.Neurons[1].Weights.Add(net.inputlayer.Neurons[0], l[3]);
            net.hiddenlayer.Neurons[1].Weights.Add(net.inputlayer.Neurons[1], l[4]);
            net.hiddenlayer.Neurons[1].Weights.Add(net.inputlayer.Neurons[2], l[5]);

            net.outputlayer.Neurons.Add(new Neuron());

            net.outputlayer.Neurons[0].Weights.Add(net.hiddenlayer.Neurons[0], l[6]);
            net.outputlayer.Neurons[0].Weights.Add(net.hiddenlayer.Neurons[1], l[7]);
            net.outputlayer.Neurons[0].Weights.Add(net.hiddenlayer.Neurons[2], l[8]);
        }

        private double nextWeight(Random rand)
        {
            return rand.NextDouble() * 0.6 - 0.3;
            //return rand.NextDouble() * 20 - 10;
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
            return Math.Pow((t - this.outputlayer.Neurons[0].Result),2.0) < 0.1;
        }

        public double checkFunction(Dictionary<double[], double> Data, List<double> weights)
        {
            double sum = 0;
            foreach (var x in Data)
            {
                this.initialize(x.Key[0], x.Key[1], weights);
                this.Run();
                sum += Math.Pow(this.outputlayer.Neurons[0].Result - x.Value,2.0);
            }
            return sum/4;
        }

        public List<double> SetWeights(Random rand)
        {
            List<double> l = new List<double>();
            for (int i = 0; i < 9; i++)
            {
                l.Add(nextWeight(rand));
            }
            return l;
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
            foreach (var x in this.outputlayer.Neurons[0].Weights)
            {
                l.Add(x.Value);
            }
            return l;
        }
    }  

    class Program
    {
        public string DoubleToStr(double x,int n1, int n2)
        {
            string s ="";
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

        public string IntToBit(double x,int n)
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
        
        public List <string> Crossing(string s1,string s2)
        {
            List<string> l = new List<string>();
            string t1 = "";
            string t2 = "";
            int n = s1.Length;
            Random rand = new Random(DateTime.Now.Millisecond);
            int temp = rand.Next(1,n);
            t1 = s1.Substring(0,temp+1)+s2.Substring(temp+1,n-temp-1);
            t2 = s2.Substring(0, temp + 1) + s1.Substring(temp + 1, n - temp - 1);
            //s1 = t1;
            //s2 = t2;
            l.Add(t1);
            l.Add(t2);
            return l;
        }

        public string Mutation(string s)
        {
            string res = "";
            string[] a = s.Split('.');
            Random rand = new Random(DateTime.Now.Millisecond);
            res += a[0][0];
            for (int i = 1; i < a[0].Length; i++)
            { 
                if (rand.Next(0,9) == 1)
                {                   
                    if (a[0][i] == '0')
                        res += '1';
                    else res += '0';                   
                }
                else res += a[0][i];
            }
           
            if (a.Length > 1)
            {
                res += '.';
                for (int i = 0; i < a[1].Length; i++)
                {
                    if (rand.Next(0, 9) == 1)
                    {
                        if (a[1][i] == '0')
                            res += '1';
                        else res += '0';
                    }
                    else res += a[1][i];
                }
            }
            return res;
        }

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
                    List<double> weights = xornet.GetWeights();
                    xornet.inputlayer.Neurons[0].Result = x.Key[0];
                    xornet.inputlayer.Neurons[1].Result = x.Key[1];
                    xornet.Run();
                    if (xornet.checkResult(x.Value))
                    {
                        continue;
                    } 
                    stop = false;
                    xornet.train(x.Value, 0.3);
                    Console.WriteLine( iterations);
                    ++iterations;
                }
            }
            Console.WriteLine("Training completed: {0} iterations", iterations);
            foreach (var x in Data)
            {
                xornet.inputlayer.Neurons[0].Result = x.Key[0];
                xornet.inputlayer.Neurons[1].Result = x.Key[1];
                xornet.Run();
                Console.WriteLine("{0} xor {1} = {2}", x.Key[0], x.Key[1], x.Value);
            }
        }

        static void Main1(string[] args)
        {
            Program pr = new Program();
           // Console.WriteLine(pr.StrToDouble(pr.DoubleToStr(-155, 10, 15)));

            Dictionary<double[], double> data = new Dictionary<double[], double>();
            List<List<double>> population = new List<List<double>>();
            data.Add(new double[] { 0, 0 }, 0);
            data.Add(new double[] { 0, 1 }, 1);
            data.Add(new double[] { 1, 0 }, 1);
            data.Add(new double[] { 1, 1 }, 0);
            Random rand = new Random();
            int n = 40;
            NeuralNetwork xornet = new NeuralNetwork();
            for (int i = 0; i < n; i++)
            {
                population.Add(xornet.SetWeights(rand));
            }
            bool stop = false;
            int number = 0;
            int iterations = 0;
            double func = 1;
            List<int> numbers1 = new List<int>();
            List<int> numbers2 = new List<int>();
            for (int i = 0; i < n; i++)
            {
                numbers1.Add(i);
            }
            List<int> numbers3 = new List<int>();
            List<int> numbers4 = new List<int>();
            for (int i = 0; i < n / 2; i++)
            {
                numbers3.Add(i);
            }
            while (!stop)
            {
                for (int i = 0; i < n; i++)
                {
                    if (xornet.checkFunction(data, population[i]) < func)
                    {
                        func = xornet.checkFunction(data, population[i]);
                        Console.WriteLine(func);
                    }
                    if (xornet.checkFunction(data, population[i]) < 0.1)
                    {
                        stop = true;
                        number = i;
                        break;
                    }
                }
                if (stop == true)
                    break;
                List<List<double>> selection = new List<List<double>>();
                /*    for (int i = 0; i < n; i += 2)
                    {
                        if (xornet.checkFunction(data, population[i]) < xornet.checkFunction(data, population[i + 1]))
                            selection.Add(population[i]);
                        else selection.Add(population[i + 1]);
                    }*/
                while (numbers1.Count() != 0)
                {
                    int k1 = numbers1[rand.Next(0, numbers1.Count())];
                    while (numbers2.Contains(k1) != false)
                    {
                        k1 = numbers1[rand.Next(0, numbers1.Count())];
                    }
                    numbers1.Remove(k1);
                    numbers2.Add(k1);

                    int k2 = numbers1[rand.Next(0, numbers1.Count())];
                    while (numbers2.Contains(k2) != false)
                    {
                        k2 = numbers1[rand.Next(0, numbers1.Count())];
                    }
                    numbers1.Remove(k2);
                    numbers2.Add(k2);

                    if (xornet.checkFunction(data, population[k1]) < xornet.checkFunction(data, population[k2]))
                        selection.Add(population[k1]);
                    else selection.Add(population[k2]);
                }
                foreach (int t in numbers2)
                    numbers1.Add(t);
                numbers2.Clear();

                List<List<double>> crossing = selection;

                /*
                for (int i = 0; i < n / 2; i += 2)
                {
                    for (int j = 0; j < population[0].Count; j++)
                    {
                        List<string> l = pr.Crossing(pr.DoubleToStr(crossing[i][j], 8, 10), pr.DoubleToStr(crossing[i + 1][j], 8, 10));
                        crossing[i][j] = pr.StrToDouble(l[0]);
                        crossing[i + 1][j] = pr.StrToDouble(l[1]);
                    }
                }*/

                while (numbers3.Count() != 0)
                {
                    int k1 = numbers3[rand.Next(0, numbers3.Count())];
                    while (numbers4.Contains(k1) != false)
                    {
                        k1 = numbers3[rand.Next(0, numbers3.Count())];
                    }
                    numbers3.Remove(k1);
                    numbers4.Add(k1);

                    int k2 = numbers3[rand.Next(0, numbers3.Count())];
                    while (numbers4.Contains(k2) != false)
                    {
                        k2 = numbers3[rand.Next(0, numbers3.Count())];
                    }
                    numbers3.Remove(k2);
                    numbers4.Add(k2);

                    for (int j = 0; j < population[0].Count; j++)
                    {
                        List<string> l = pr.Crossing(pr.DoubleToStr(crossing[k1][j], 8, 10), pr.DoubleToStr(crossing[k2][j], 8, 10));
                        crossing[k1][j] = pr.StrToDouble(l[0]);
                        crossing[k2][j] = pr.StrToDouble(l[1]);
                    }
                }
                foreach (int t in numbers3)
                    numbers3.Add(t);
                numbers4.Clear();


                for (int i = 0; i < n / 2; i++)
                {
                    for (int j = 0; j < 9; j++)
                    {
                        crossing[i][j] = pr.StrToDouble(pr.Mutation(pr.DoubleToStr(crossing[i][j], 8, 10)));
                        selection[i][j] = pr.StrToDouble(pr.Mutation(pr.DoubleToStr(selection[i][j], 10, 10)));
                    }
                }

                population.Clear();
                foreach (var t in selection)
                    population.Add(t);
                foreach (var t in crossing)
                    population.Add(t);
                iterations++;
            }
            Console.WriteLine("Training completed: {0} iterations", iterations);
            foreach (var x in data)
            {
                xornet.inputlayer.Neurons[0].Result = x.Key[0];
                xornet.inputlayer.Neurons[1].Result = x.Key[1];
                xornet.Run();
                Console.WriteLine("{0} xor {1} = {2} ({3})", x.Key[0], x.Key[1], xornet.OutputLayer.Neurons[0].Result, x.Value);
            }
        }
    }
}
