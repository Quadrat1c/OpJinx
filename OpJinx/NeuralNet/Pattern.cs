using System;

namespace OpJinx
{
    public class Pattern
    {
        private double[] _inputs;
        private double _output;

        public Pattern(string value, int inputSize)
        {
            string[] line = value.Split(',');

            if (line.Length - 1 != inputSize)
                throw new Exception("Input does not match neural network configuration");

            _inputs = new double[inputSize];

            for (int i = 0; i < inputSize; i++)
            {
                _inputs[i] = double.Parse(line[i]);
            }

            _output = double.Parse(line[inputSize]);
        }

        public double[] Inputs
        {
            get { return _inputs; }
        }

        public double Output
        {
            get { return _output; }
        }
    }
}
