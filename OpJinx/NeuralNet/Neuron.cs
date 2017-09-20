using System;
using System.Collections.Generic;

namespace OpJinx
{
    public class Neuron
    {
        private double _bias;                       // Bias value.
        private double _error;                      // Sum of error.
        private double _input;                      // Sum of inputs.
        private double _lambda = 6;                 // Steepness of sigmoid curve.
        private double _learnRate = 0.5;            // Learning rate.
        private double _output = double.MinValue;   // Preset value of neuron.
        private List<Weight> _weights;              // Collection of weights to inputs.

        public Neuron() { }

        public Neuron(Layer inputs, Random rnd)
        {
            _weights = new List<Weight>();

            foreach (Neuron input in inputs)
            {
                Weight w = new Weight();
                w.Input = input;
                w.Value = rnd.NextDouble() * 2 - 1;
                _weights.Add(w);
            }
        }

        public void Activate()
        {
            _input = 0;

            foreach (Weight w in _weights)
            {
                _input += w.Value * w.Input.Output;
            }
        }

        public double ErrorFeedback(Neuron input)
        {
            Weight w = _weights.Find(delegate (Weight t) { return t.Input == input; });
            return _error * Derivative * w.Value;
        }

        public void AdjustWeights(double value)
        {
            _error = value;

            for (int i = 0; i < _weights.Count; i++)
            {
                _weights[i].Value += _error * Derivative * _learnRate * _weights[i].Input.Output;
            }

            _bias += _error * Derivative * _learnRate;
        }

        private double Derivative
        {
            get
            {
                double activation = Output;
                return activation * (1 - activation);
            }
        }

        public double Output
        {
            get
            {
                if (_output != double.MinValue)
                {
                    return _output;
                }

                return 1 / (1 + Math.Exp(-_lambda * (_input + _bias)));
            }
            set
            {
                _output = value;
            }
        }
    }
}