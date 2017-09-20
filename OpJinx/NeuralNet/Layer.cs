using System;
using System.Collections.Generic;

namespace OpJinx
{
    public class Layer : List<Neuron>
    {
        public Layer(int size)
        {
            for (int i = 0; i < size; i++)
                base.Add(new Neuron());
        }

        public Layer(int size, Layer layer, Random rnd)
        {
            for (int i = 0; i < size; i++)
                base.Add(new Neuron(layer, rnd));
        }
    }
}