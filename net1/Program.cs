using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace net1
{
    class Program
    {
  
        public interface INeuronReceptor
        {
            Dictionary<INeuronSignal, double> Input { get; }
        }

        public interface INeuronSignal
        {
            double Output { get; set; }
        }

        public interface INeuron : INeuronSignal, INeuronReceptor
        {
            void Pulse(INeuralLayer layer);
            double Bias { get; set; }
        }

        public interface INeuralLayer : IList<INeuron>
        {
            void Pulse(INeuralNet net);
        }

        public interface INeuralNet
        {
            INeuralLayer InputLayer { get; }
            INeuralLayer HiddenLayer { get; }
            INeuralLayer OutputLayer { get; }
            void Pulse();
        }

        public class Neuron : INeuron
        {
            public Neuron(double bias,double weight)
            {
                m_bias = bias;
                m_bias_weight = weight;
                m_input = new Dictionary<INeuronSignal, double>();
            }

            private Dictionary<INeuronSignal, double> m_input;
            double m_output;
            double m_bias;
            double m_bias_weight;
          
            public double Output
            {
                get { return m_output; }
                set { m_output = value; }
            }

            public Dictionary<INeuronSignal, double> Input
            {
                get { return m_input; }
            }

          
            public void Pulse(INeuralLayer layer)
            {
                m_output = 0;
                foreach (KeyValuePair<INeuronSignal, double> item in m_input)
                    m_output += item.Key.Output * item.Value;
                m_output += m_bias * m_bias_weight;
                m_output = Sigmoid(m_output);
            }

            public double Bias
            {
                get { return m_bias; }
                set { m_bias = value; }
            }

            public static double Sigmoid(double value)
            {
                return (1 / (1 + Math.Exp(-value)) > 0.6) ? 1 : 0;     
            }
        }

        public class NeuralLayer : INeuralLayer
        {

            public NeuralLayer()
            {
                m_neurons = new List<INeuron>();
            }

            private List<INeuron> m_neurons;

            public int IndexOf(INeuron item)
            {
                return m_neurons.IndexOf(item);
            }

            public void Insert(int index, INeuron item)
            {
                m_neurons.Insert(index, item);
            }

            public void RemoveAt(int index)
            {
                m_neurons.RemoveAt(index);
            }

            public INeuron this[int index]
            {
                get { return m_neurons[index]; }
                set { m_neurons[index] = value; }
            }

            public void Add(INeuron item)
            {
                m_neurons.Add(item);
            }

            public void Clear()
            {
                m_neurons.Clear();
            }

            public bool Contains(INeuron item)
            {
                return m_neurons.Contains(item);
            }

            public void CopyTo(INeuron[] array, int arrayIndex)
            {
                m_neurons.CopyTo(array, arrayIndex);
            }

            public int Count
            {
                get { return m_neurons.Count; }
            }

            public bool IsReadOnly
            {
                get { return false; }
            }

            public bool Remove(INeuron item)
            {
                return m_neurons.Remove(item);
            }

            public IEnumerator<INeuron> GetEnumerator()
            {
                return m_neurons.GetEnumerator();
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }

            public void Pulse(INeuralNet net)
            {
                foreach (INeuron n in m_neurons)
                    n.Pulse(this);
            }
          
        }

        public class NeuralNet : INeuralNet
        {

            private INeuralLayer m_inputLayer;
            private INeuralLayer m_outputLayer;
            private INeuralLayer m_hiddenLayer;

            public INeuralLayer InputLayer
            {
                get { return m_inputLayer; }
            }

            public INeuralLayer HiddenLayer
            {
                get { return m_hiddenLayer; }
            }

            public INeuralLayer OutputLayer
            {
                get { return m_outputLayer; }
            }

            public void Pulse()
            {
                m_hiddenLayer.Pulse(this);
                m_outputLayer.Pulse(this);
            }          

            public void Initialize(double x1, double x2)
            {
                Initialize(this, x1, x2);
            }           

            private static void Initialize(NeuralNet net, double x1, double x2)            
            {
              
                net.m_inputLayer = new NeuralLayer();
                net.m_outputLayer = new NeuralLayer();
                net.m_hiddenLayer = new NeuralLayer();

                net.m_inputLayer.Add(new Neuron(0,0));
                net.m_inputLayer.Add(new Neuron(0,0));

                net.m_inputLayer[0].Output = x1;
                net.m_inputLayer[1].Output = x2;

                net.m_hiddenLayer.Add(new Neuron(1, 1.5));
                net.m_hiddenLayer.Add(new Neuron(1, 0.5));

                net.m_outputLayer.Add(new Neuron(1,-0.5));
                    
                net.m_hiddenLayer[0].Input.Add(net.m_inputLayer[0], -1);
                net.m_hiddenLayer[0].Input.Add(net.m_inputLayer[1], -1);
                net.m_hiddenLayer[1].Input.Add(net.m_inputLayer[0], -1);
                net.m_hiddenLayer[1].Input.Add(net.m_inputLayer[1], -1);
                 
                net.m_outputLayer[0].Input.Add(net.m_hiddenLayer[0], 1);
                net.m_outputLayer[0].Input.Add(net.m_hiddenLayer[1], -1);
            }

        }

        static void Main(string[] args)
        {
            NeuralNet xornet = new NeuralNet();
            for (int i = 0; i < 2;++i)
            {
                for (int j = 0; j < 2;++j)
                {
                    xornet.Initialize(i, j);
                    xornet.Pulse();
                    Console.WriteLine("{0} xor {1} = {2}", i, j, xornet.OutputLayer[0].Output);
                }
            }
        }
    }
}
