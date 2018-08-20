using System;
using System.IO;
using System.Net;
using System.Threading;
using System.Collections.Generic;


namespace JinxNeuralNetwork
{
    //neural network class
    /// <summary>
    /// Neural network.
    /// </summary>
    public class NeuralNetwork
    {

        public NeuralNetworkLayer inputLayer;
        public NeuralNetworkLayer outputLayer;
        public NeuralNetworkLayer[] hiddenLayers;

        public NeuralNetworkLayerConnection outputConnection;
        public NeuralNetworkLayerConnection[] hiddenConnections, hiddenRecurringConnections;

        public int maxNumberOfHiddenNeurons, maxNumberOfSynapses;

        /// <summary>
        /// Create new NeuralNetwork from existing(src).
        /// </summary>
        /// <param name="src">Existing NeuralNetwork to copy.</param>
        public NeuralNetwork(NeuralNetwork src)
        {
            inputLayer = new NeuralNetworkLayer(src.inputLayer);

            hiddenLayers = new NeuralNetworkLayer[src.hiddenLayers.Length];
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i] = new NeuralNetworkLayer(src.hiddenLayers[i]);
                hiddenLayers[i].Init();
            }
            outputLayer = new NeuralNetworkLayer(src.outputLayer);
            outputLayer.Init();

            //setup layer connections
            if (hiddenLayers.Length > 0)
            {
                //hidden layer connections
                hiddenConnections = new NeuralNetworkLayerConnection[hiddenLayers.Length];
                hiddenRecurringConnections = new NeuralNetworkLayerConnection[hiddenLayers.Length];
                maxNumberOfHiddenNeurons = 0;
                maxNumberOfSynapses = 0;
                for (int i = 0; i < hiddenLayers.Length; i++)
                {
                    if (i == 0) hiddenConnections[0] = new NeuralNetworkLayerConnection(inputLayer, hiddenLayers[0]);
                    else hiddenConnections[i] = new NeuralNetworkLayerConnection(hiddenLayers[i - 1], hiddenLayers[i]);
                    //recurrent connection for hidden layer
                    if (hiddenLayers[i].recurring)
                    {
                        hiddenRecurringConnections[i] = new NeuralNetworkLayerConnection(hiddenLayers[i], hiddenLayers[i]);
                    }
                    else
                    {
                        hiddenRecurringConnections[i] = null;
                    }
                    //calc max hidden neurons
                    if (hiddenLayers[i].numberOfNeurons > maxNumberOfHiddenNeurons)
                    {
                        maxNumberOfHiddenNeurons = hiddenLayers[i].numberOfNeurons;
                    }
                    if (hiddenConnections[i].numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = hiddenConnections[i].numberOfSynapses;
                }

                //output connection
                outputConnection = new NeuralNetworkLayerConnection(hiddenLayers[hiddenLayers.Length - 1], outputLayer);

                if (outputConnection.numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = outputConnection.numberOfSynapses;
            }
            else
            {
                maxNumberOfHiddenNeurons = 0;
                maxNumberOfSynapses = 0;

                //direct input to output connection
                outputConnection = new NeuralNetworkLayerConnection(inputLayer, outputLayer);
                if (outputConnection.numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = outputConnection.numberOfSynapses;

            }
        }
        /// <summary>
        /// Create new NeuralNetwork.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="hidden"></param>
        /// <param name="output"></param>
        public NeuralNetwork(NeuralNetworkLayer input, NeuralNetworkLayer[] hidden, NeuralNetworkLayer output)
        {

            inputLayer = input;
            hiddenLayers = hidden;
            outputLayer = output;
            outputLayer.Init();

            //setup layer connections
            if (hidden.Length > 0)
            {
                //hidden layer connections
                hiddenConnections = new NeuralNetworkLayerConnection[hidden.Length];
                hiddenRecurringConnections = new NeuralNetworkLayerConnection[hidden.Length];
                maxNumberOfHiddenNeurons = 0;
                maxNumberOfSynapses = 0;
                for (int i = 0; i < hidden.Length; i++)
                {
                    if (i == 0) hiddenConnections[0] = new NeuralNetworkLayerConnection(input, hidden[0]);
                    else hiddenConnections[i] = new NeuralNetworkLayerConnection(hidden[i - 1], hidden[i]);

                    hiddenLayers[i].Init();

                    //recurrent connection for hidden layer
                    if (hidden[i].recurring)
                    {
                        hiddenRecurringConnections[i] = new NeuralNetworkLayerConnection(hidden[i], hidden[i]);
                    }
                    else
                    {
                        hiddenRecurringConnections[i] = null;
                    }
                    //calc max hidden neurons
                    if (hidden[i].numberOfNeurons > maxNumberOfHiddenNeurons)
                    {
                        maxNumberOfHiddenNeurons = hidden[i].numberOfNeurons;
                    }
                    if (hiddenConnections[i].numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = hiddenConnections[i].numberOfSynapses;

                }

                //output connection
                outputConnection = new NeuralNetworkLayerConnection(hidden[hidden.Length - 1], output);
                if (outputConnection.numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = outputConnection.numberOfSynapses;
            }
            else
            {
                maxNumberOfHiddenNeurons = 0;
                maxNumberOfSynapses = 0;

                //direct input to output connection
                outputConnection = new NeuralNetworkLayerConnection(input, output);
                if (outputConnection.numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = outputConnection.numberOfSynapses;
            }
        }

        //execute neural network
        /// <summary>
        /// Execute NeuralNetwork.
        /// </summary>
        /// <param name="context">Execution memory.</param>
        public void Execute(NeuralNetworkContext context)
        {
            float[] input = context.inputData,
                    output = context.outputData,
                    hidden = context.hiddenData;
            float[][] hiddenRecurring = context.hiddenRecurringData;

            int i, weightIndex, recurringWeightIndex;


            NeuronActivationFunction activeFunc;
            if (hiddenLayers.Length > 0)
            {
                int lastNumNeurons = 0;
                float[] weights, biases, recurringWeights;
                for (i = 0; i < hiddenLayers.Length; i++)
                {

                    weights = hiddenConnections[i].weights;
                    biases = hiddenLayers[i].biases;

                    activeFunc = hiddenLayers[i].activationFunction;

                    float[] ina;
                    int alen;
                    if (i == 0)
                    {
                        ina = input;
                        alen = input.Length;
                    }
                    else
                    {
                        ina = hidden;
                        alen = lastNumNeurons;
                    }

                    if (hiddenLayers[i].recurring)
                    {
                        //recurring hidden layer
                        recurringWeights = hiddenRecurringConnections[i].weights;

                        weightIndex = 0;
                        recurringWeightIndex = 0;

                        float[] hrec = hiddenRecurring[i];

                        int k = biases.Length;
                        while (k-- > 0)
                        {
                            float ov = biases[k];

                            int j = alen;
                            while (j-- > 0)
                            {
                                ov += ina[j] * weights[weightIndex++];
                            }

                            j = hrec.Length;
                            while (j-- > 0)
                            {
                                ov += hrec[j] * recurringWeights[recurringWeightIndex++];
                            }

                            hidden[k] = activeFunc(ov);
                        }

                        Array.Copy(hidden, hrec, biases.Length);
                    }
                    else
                    {
                        //non recurring hidden layer
                        weightIndex = 0;

                        int k = biases.Length;
                        while (k-- > 0)
                        {
                            float ov = biases[k];

                            int j = alen;
                            while (j-- > 0)
                            {
                                ov += ina[j] * weights[weightIndex++];
                            }

                            hidden[k] = activeFunc(ov);
                        }
                    }

                    lastNumNeurons = biases.Length;
                }

                activeFunc = outputLayer.activationFunction;

                //last output layer

                //run input to output layer connection
                weights = outputConnection.weights;
                biases = outputLayer.biases;

                weightIndex = 0;
                recurringWeightIndex = 0;

                i = output.Length;
                while (i-- > 0)
                {
                    float ov = biases[i];

                    //input connections
                    int k = lastNumNeurons;
                    while (k-- > 0)
                    {
                        ov += hidden[k] * weights[weightIndex++];
                    }

                    output[i] = activeFunc(ov);
                }
            }
            else
            {
                activeFunc = outputLayer.activationFunction;

                //run input to output layer connection with recurring output
                float[] weights = outputConnection.weights,
                        biases = outputLayer.biases;

                weightIndex = 0;
                recurringWeightIndex = 0;
                i = output.Length;
                while (i-- > 0)
                {
                    float ov = biases[i];

                    //input connections
                    int k = input.Length;
                    while (k-- > 0)
                    {
                        ov += input[k] * weights[weightIndex++];
                    }

                    output[i] = activeFunc(ov);
                }
            }
        }

        //execute neural network and save all calculation results in fullContext for adagrad
        /// <summary>
        /// Execute neural network and save all calculation results in fullContext for adagrad
        /// </summary>
        /// <param name="input"></param>
        /// <param name="context"></param>
        /// <param name="fullContext"></param>
        public void Execute_FullContext(NeuralNetworkContext context, NeuralNetworkFullContext fullContext)
        {
            float[] input = context.inputData,
                    output = context.outputData,
                    hidden = context.hiddenData;

            float[][] hiddenRecurring = context.hiddenRecurringData;

            int i, weightIndex, recurringWeightIndex;


            NeuronActivationFunction activeFunc;
            if (hiddenLayers.Length > 0)
            {
                int lastNumNeurons = 0;
                float[] weights, biases, recurringWeights;
                for (i = 0; i < hiddenLayers.Length; i++)
                {
                    weights = hiddenConnections[i].weights;
                    biases = hiddenLayers[i].biases;

                    activeFunc = hiddenLayers[i].activationFunction;

                    float[] ina;
                    int alen;
                    if (i == 0)
                    {
                        ina = input;
                        alen = input.Length;
                    }
                    else
                    {
                        ina = hidden;
                        alen = lastNumNeurons;
                    }

                    if (hiddenLayers[i].recurring)
                    {
                        //recurring hidden layer
                        float[] hrec = hiddenRecurring[i];

                        recurringWeights = hiddenRecurringConnections[i].weights;

                        //copy over data needed for training
                        Array.Copy(hrec, fullContext.hiddenRecurringBuffer[i], hrec.Length);

                        weightIndex = 0;
                        recurringWeightIndex = 0;


                        int k = biases.Length;
                        while (k-- > 0)
                        {
                            float ov = biases[k];

                            int j = alen;
                            while (j-- > 0)
                            {
                                ov += ina[j] * weights[weightIndex++];
                            }

                            j = hrec.Length;
                            while (j-- > 0)
                            {
                                ov += hrec[j] * recurringWeights[recurringWeightIndex++];
                            }

                            hidden[k] = activeFunc(ov);
                        }

                        Array.Copy(hidden, hrec, biases.Length);
                    }
                    else
                    {
                        //non recurring hidden layer
                        weightIndex = 0;

                        int k = biases.Length;
                        while (k-- > 0)
                        {
                            float ov = biases[k];

                            int j = alen;
                            while (j-- > 0)
                            {
                                ov += ina[j] * weights[weightIndex++];
                            }

                            hidden[k] = activeFunc(ov);
                        }
                    }

                    Array.Copy(hidden, fullContext.hiddenBuffer[i], biases.Length);
                    lastNumNeurons = biases.Length;
                }

                activeFunc = outputLayer.activationFunction;


                //last output layer

                //run input to output layer connection
                weights = outputConnection.weights;
                biases = outputLayer.biases;

                weightIndex = 0;
                recurringWeightIndex = 0;

                i = output.Length;
                while (i-- > 0)
                {
                    float ov = biases[i];

                    //input connections
                    int k = lastNumNeurons;
                    while (k-- > 0)
                    {
                        ov += hidden[k] * weights[weightIndex++];
                    }

                    output[i] = activeFunc(ov);
                }
            }
            else
            {
                activeFunc = outputLayer.activationFunction;

                //run input to output layer connection with recurring output
                float[] weights = outputConnection.weights,
                        biases = outputLayer.biases;


                weightIndex = 0;
                recurringWeightIndex = 0;
                i = output.Length;
                while (i-- > 0)
                {
                    float ov = biases[i];

                    //input connections
                    int k = input.Length;
                    while (k-- > 0)
                    {
                        ov += input[k] * weights[weightIndex++];
                    }

                    output[i] = activeFunc(ov);
                }
            }
        }



        /// <summary>
        /// Run neural network backwards calculating derivatives to use for adagrad or generation.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="context"></param>
        /// <param name="fullContext"></param>
        /// <param name="derivMem"></param>
        public void ExecuteBackwards(float[] target, NeuralNetworkContext context, NeuralNetworkFullContext fullContext, NeuralNetworkPropagationState propState, int lossType, int crossEntropyTarget)
        {
            //prepare for back propagation
            for (int i = 0; i < propState.state.Length; i++) {
                Utils.Fill(propState.state[i], 0.0f);
            }

            //back propagation + calculate max loss
            int lid = hiddenLayers.Length;

            float lossAvg = 0.0f;
            for (int i = 0; i < target.Length; i++)
            {
                float deriv = context.outputData[i] - target[i];

                if (lossType == NeuralNetworkTrainer.LOSS_TYPE_MAX)
                {
                    float aderiv = Math.Abs(deriv);
                    if (aderiv > lossAvg) lossAvg = aderiv;
                }
                else if (lossType == NeuralNetworkTrainer.LOSS_TYPE_AVERAGE)
                {
                    lossAvg += Math.Abs(deriv);
                }

                backpropagate(lid, i, deriv, propState);
            }

            if (lossType == NeuralNetworkTrainer.LOSS_TYPE_AVERAGE)
            {
                lossAvg /= (float)target.Length;
            }
            else
            {
                if (lossType == NeuralNetworkTrainer.LOSS_TYPE_CROSSENTROPY && crossEntropyTarget != -1)
                {
                    lossAvg = (float)-Math.Log(context.outputData[crossEntropyTarget]);
                    if (float.IsInfinity(lossAvg))
                    {
                        lossAvg = 1e8f;
                    }
                }
            }

            propState.loss = lossAvg;
            propState.derivativeMemory.SwapBPBuffers();


            int k = lid;
            while (k-- > 0)
            {
                int l = hiddenLayers[k].numberOfNeurons;
                while (l-- > 0)
                {
                    backpropagate(k, l, propState.state[k][l], propState);
                }
            }
        }

        private void backpropagate(int level, int index, float deriv, NeuralNetworkPropagationState propState)
        {
            if (level < 0) return;

            int i, weightIndex;
            float[] b, m, w;


            //recurring weights
            if (level < propState.recurrWeightMems.Length && propState.recurrWeightMems[level] != null)
            {
                b = propState.recurrBuf[level];
                m = propState.recurrWeightMems[level];
                w = propState.recurrWeights[level];

                i = b.Length;
                weightIndex = w.Length - (index + 1) * i;
                float nhderiv = 0.0f;
                while (i-- > 0)
                {
                    m[weightIndex] += deriv * b[i];
                    nhderiv += deriv * w[weightIndex];
                    weightIndex++;
                }

#pragma warning disable 414,1718
                if (nhderiv != nhderiv || float.IsInfinity(nhderiv))
                {
                    nhderiv = 0.0f;
                }
#pragma warning restore 1718
                propState.derivativeMemory.altRecurringBPBuffer[level][index] = nhderiv;
            }


            float[] bpb = null;

            //biases and weights
            b = propState.buf[level];
            m = propState.weightMems[level];
            w = propState.weights[level];

            bpb = null;
            if (level != 0) bpb = propState.derivativeMemory.recurringBPBuffer[level - 1];

            propState.biasMems[level][index] += deriv;

            i = b.Length;
            weightIndex = w.Length - (index + 1) * i;
            while (i-- > 0)
            {
                float nderiv = b[i];
                m[weightIndex] += deriv * nderiv;
                if (level != 0)
                {
                    nderiv *= nderiv;

                    float bpropderiv = 0.0f;
                    if (bpb != null)
                    {
                        bpropderiv = bpb[i];
                    }

                    propState.state[level - 1][i] += (1.0f - nderiv) * (deriv * w[weightIndex] + bpropderiv);
                }
                else
                {
                    if (propState.inputMem != null)
                    {
                        nderiv *= nderiv;

                        float bpropderiv = 0.0f;
                        if (bpb != null)
                        {
                            bpropderiv = bpb[i];
                        }

                        nderiv = (1.0f - nderiv) * (deriv * w[weightIndex] + bpropderiv);
                        propState.inputMem[i] += nderiv;
                    }
                }
                weightIndex++;
            }
        }





        //breed data with partner
        //partPartner is a value from 0 - 1 indicating the % of data to take from partner, 0 being no data and 1 being 100% partner data
        /// <summary>
        /// Breed weights/biases with partner.
        /// </summary>
        /// <param name="partner"></param>
        public void Breed(NeuralNetwork partner)
        {
            outputLayer.Breed(partner.outputLayer);

            outputConnection.Breed(partner.outputConnection);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].Breed(partner.hiddenLayers[i]);
                hiddenConnections[i].Breed(partner.hiddenConnections[i]);
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].Breed(partner.hiddenRecurringConnections[i]);
            }
        }

        //mutate weights and biases
        /// <summary>
        /// Mutate weights and biases.
        /// </summary>
        /// <param name="selectionChance">Chance(0-1) of a weight/bias being mutated.</param>
        public void Mutate(float selectionChance)
        {
            outputLayer.Mutate(selectionChance);
            outputConnection.Mutate(selectionChance);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].Mutate(selectionChance);
                hiddenConnections[i].Mutate(selectionChance);
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].Mutate(selectionChance);
            }

        }


        /// <summary>
        /// Randomize weights and biases specifically for adagrad.
        /// </summary>
        public void RandomizeWeightsAndBiasesForAdagrad()
        {
            NeuralNetworkLayer.MIN_BIAS = 0.0f;
            NeuralNetworkLayer.MAX_BIAS = 0.0f;
            NeuralNetworkLayerConnection.MIN_WEIGHT = 0.0f;
            NeuralNetworkLayerConnection.MAX_WEIGHT = 1.0f / maxNumberOfHiddenNeurons;
            RandomizeWeightsAndBiases();
        }

        //randomize weights and biases of layers and connections
        /// <summary>
        /// Randomize all weights/biases.
        /// </summary>
        public void RandomizeWeightsAndBiases()
        {
            outputLayer.RandomizeBiases();
            outputConnection.RandomizeWeights();

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].RandomizeBiases();
                hiddenConnections[i].RandomizeWeights();
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].RandomizeWeights();
            }

        }

        /// <summary>
        /// Randomize all weights and biases between specified min/max values.
        /// </summary>
        public void RandomizeWeightsAndBiases(float minBias, float maxBias, float minWeight, float maxWeight)
        {
            NeuralNetworkLayer.MIN_BIAS = minBias;
            NeuralNetworkLayer.MAX_BIAS = maxBias;
            NeuralNetworkLayerConnection.MIN_WEIGHT = minWeight;
            NeuralNetworkLayerConnection.MAX_WEIGHT = maxWeight;

            outputLayer.RandomizeBiases();
            outputConnection.RandomizeWeights();

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].RandomizeBiases();
                hiddenConnections[i].RandomizeWeights();
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].RandomizeWeights();
            }

        }

        //copy weights and biases from another neural network
        /// <summary>
        /// Copy weights and biases from another NeuralNetwork(nn).
        /// </summary>
        /// <param name="nn"></param>
        public void CopyWeightsAndBiases(NeuralNetwork nn)
        {
            outputLayer.CopyBiases(nn.outputLayer);
            outputConnection.CopyWeights(nn.outputConnection);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].CopyBiases(nn.hiddenLayers[i]);
                hiddenConnections[i].CopyWeights(nn.hiddenConnections[i]);
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].CopyWeights(nn.hiddenRecurringConnections[i]);
            }

        }

        /// <summary>
        /// Setup data arrays for execution.
        /// </summary>
        /// <param name="ina"></param>
        /// <param name="outa"></param>
        /// <param name="hiddena"></param>
        /// <param name="hiddenRecurra"></param>
        public void SetupExecutionArrays(out float[] ina, out float[] outa, out float[] hiddena, out float[][] hiddenRecurra)
        {
            ina = new float[inputLayer.numberOfNeurons];
            outa = new float[outputLayer.numberOfNeurons];
            hiddena = new float[maxNumberOfHiddenNeurons];

            hiddenRecurra = new float[hiddenLayers.Length][];
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                if (hiddenLayers[i].recurring)
                {
                    hiddenRecurra[i] = new float[hiddenLayers[i].numberOfNeurons];
                }
            }
        }

        /// <summary>
        /// Save structure of NeuralNetwork layers to stream.
        /// </summary>
        /// <param name="s"></param>
        public void SaveStructure(Stream s)
        {
            inputLayer.SaveStructure(s);

            Utils.IntToStream(hiddenLayers.Length, s);
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].SaveStructure(s);
            }
            outputLayer.SaveStructure(s);
        }

        /// <summary>
        /// Load NeuralNetwork structure from stream.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static NeuralNetwork LoadStructure(Stream s)
        {
            NeuralNetworkLayer inLayer = new NeuralNetworkLayer();
            inLayer.LoadStructure(s);

            NeuralNetworkLayer[] hidden = new NeuralNetworkLayer[Utils.IntFromStream(s)];
            for (int i = 0; i < hidden.Length; i++)
            {
                hidden[i] = new NeuralNetworkLayer();
                hidden[i].LoadStructure(s);
            }

            NeuralNetworkLayer outLayer = new NeuralNetworkLayer();
            outLayer.LoadStructure(s);

            return new NeuralNetwork(inLayer, hidden, outLayer);
        }

        //save data to stream
        /// <summary>
        /// Save NeuralNetwork data(weights/biases, no structure data like input/hidden/output) to stream.
        /// </summary>
        /// <param name="s"></param>
        public void Save(Stream s)
        {
            outputLayer.Save(s);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].Save(s);
            }

            outputConnection.Save(s);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenConnections[i].Save(s);
                if (hiddenLayers[i].recurring)
                {
                    hiddenRecurringConnections[i].Save(s);
                }
            }
        }

        //load data from stream
        /// <summary>
        /// Load NeuralNetwork from stream(s).
        /// </summary>
        /// <param name="s"></param>
        public void Load(Stream s)
        {
            outputLayer.Load(s);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].Load(s);
            }

            outputConnection.Load(s);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenConnections[i].Load(s);
                if (hiddenLayers[i].recurring)
                {
                    hiddenRecurringConnections[i].Load(s);
                }
            }
        }

        //get total number of neurons in network
        /// <summary>
        /// Get total number of neurons in network.
        /// </summary>
        /// <returns></returns>
        public int TotalNumberOfNeurons()
        {
            int nneurons = inputLayer.numberOfNeurons + outputLayer.numberOfNeurons;
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                nneurons += hiddenLayers[i].numberOfNeurons;
            }
            return nneurons;
        }

        //get total number of synapses in network
        /// <summary>
        /// Get total number of synapses in network.
        /// </summary>
        /// <returns></returns>
        public int TotalNumberOfSynapses()
        {
            int nsynapses = outputConnection.numberOfSynapses;
            if (hiddenConnections != null)
            {
                for (int i = 0; i < hiddenConnections.Length; i++)
                {
                    nsynapses += hiddenConnections[i].numberOfSynapses;
                    if (hiddenLayers[i].recurring) nsynapses += hiddenRecurringConnections[i].numberOfSynapses;
                }
            }
            return nsynapses;
        }


        public int NumberOfLayers()
        {
            return hiddenLayers.Length + 1;
        }

        public NeuralNetworkLayer GetLayer(int i)
        {
            if (i < hiddenLayers.Length) return hiddenLayers[i];
            return outputLayer;
        }
        public NeuralNetworkLayerConnection GetConnection(int i)
        {
            if (i < hiddenLayers.Length) return hiddenConnections[i];
            return outputConnection;
        }
        public NeuralNetworkLayerConnection GetRecurringConnection(int i)
        {
            if (i < hiddenLayers.Length) return hiddenRecurringConnections[i];
            return null;
        }

        public delegate float NeuronActivationFunction(float v);
    }
        /// <summary>
    /// Neural network layer.
    /// </summary>
    public class NeuralNetworkLayer
    {

        /// <summary>
        ///  Minimum bias when randomly generating biases.
        /// </summary>
        public static float MIN_BIAS = 0.0f;
        /// <summary>
        ///  Maximum bias when randomly generating biases.
        /// </summary>
        public static float MAX_BIAS = 0.0f;

        /// <summary>
        /// Number of neurons in layer.
        /// </summary>
        public int numberOfNeurons;
        /// <summary>
        /// Flag indicating whether or not neurons are recurring(last state is fed back in as input).
        /// </summary>
        public bool recurring;
        /// <summary>
        /// Array of biases.
        /// </summary>
        public float[] biases;
        /// <summary>
        /// Neuron activation function to use for all neurons in layer.
        /// </summary>
        public NeuralNetwork.NeuronActivationFunction activationFunction;

        /// <summary>
        /// Create new struct from existing.
        /// </summary>
        /// <param name="src">Existing layer to copy.</param>
        public NeuralNetworkLayer(NeuralNetworkLayer src)
        {
            numberOfNeurons = src.numberOfNeurons;
            recurring = src.recurring;
            activationFunction = src.activationFunction;
            biases = null;
        }
        /// <summary>
        /// Create new struct.
        /// </summary>
        /// <param name="numNeurons">Number of neurons in layer.</param>
        /// <param name="recurrin">Are neurons in layer recurring.</param>
        /// <param name="activeFunc">Neuron activation function.</param>
        public NeuralNetworkLayer(int numNeurons, bool recurrin, NeuralNetwork.NeuronActivationFunction activeFunc)
        {
            numberOfNeurons = numNeurons;
            recurring = recurrin;
            activationFunction = activeFunc;
            biases = null;
        }
        public NeuralNetworkLayer(){}

        /// <summary>
        /// Allocate biases float array.
        /// </summary>
        public void Init()
        {
            biases = new float[numberOfNeurons];
        }


        public void SaveStructure(Stream s)
        {
            Utils.IntToStream(numberOfNeurons, s);
            s.WriteByte(recurring ? (byte)1 : (byte)0);
            Utils.IntToStream(Utils.GetActivationFunctionID(activationFunction), s);
        }

        public void LoadStructure(Stream s)
        {
            numberOfNeurons = Utils.IntFromStream(s);
            recurring = s.ReadByte() == 1;
            activationFunction = Utils.GetActivationFunctionFromID(Utils.IntFromStream(s));
        }

        /// <summary>
        /// Save layer to stream.
        /// </summary>
        /// <param name="s">Stream.</param>
        public void Save(Stream s)
        {
            int i = numberOfNeurons;
            while (i-- > 0)
            {
                s.Write(Utils.FloatToBytes(biases[i]), 0, 4);
            }
        }

        /// <summary>
        /// Load layer from stream.
        /// </summary>
        /// <param name="s">Stream.</param>
        public void Load(Stream s)
        {
            byte[] rbuf = new byte[4];

            int i = numberOfNeurons;
            while (i-- > 0)
            {
                s.Read(rbuf, 0, 4);
                biases[i] = Utils.FloatFromBytes(rbuf);
            }
        }

        /// <summary>
        /// Generate random biases for layer from MIN_BIAS to MAX_BIAS.
        /// </summary>
        public void RandomizeBiases()
        {
            int i = numberOfNeurons;
            while (i-- > 0)
            {
                biases[i] = Utils.NextFloat01() * (MAX_BIAS - MIN_BIAS) + MIN_BIAS;
            }
        }

        /// <summary>
        /// Copy biases from layer.
        /// </summary>
        /// <param name="nnl">Layer.</param>
        public void CopyBiases(NeuralNetworkLayer nnl)
        {
            float[] cb = nnl.biases;

            int i = numberOfNeurons;
            while (i-- > 0)
            {
                biases[i] = cb[i];
            }
        }



        /// <summary>
        /// Mutate a selection of biases randomly.
        /// </summary>
        /// <param name="selectionChance">The chance(0-1) of a bias being mutated.</param>
        public void Mutate(float selectionChance)
        {
            int i = numberOfNeurons;
            while (i-- > 0)
            {
                if (Utils.NextFloat01() <= selectionChance)
                {
                    biases[i] = Utils.NextFloat01() * (MAX_BIAS - MIN_BIAS) + MIN_BIAS;
                }
            }
        }

        //breed data with partner, partner must have the same # neurons/synapses
        //takes a random selection of weights and biases from partner and replaces the a %(partPartner) of this classes weights/classes with those
        //partPartner is the % of weights and biases to use from the partner, 0 being none and 1 being all the weights/biases
        /// <summary>
        /// Breed with another layer(partner) taking a %(partPartner) of randomly selected biases.
        /// </summary>
        /// <param name="partner">Partner layer.</param>
        /// <param name="partPartner">Percent(0-1) of biases to take from partner.</param>
        public void Breed(NeuralNetworkLayer partner)
        {
            int i = numberOfNeurons;
            while (i-- > 0)
            {
                //randomly mix
                float val = Utils.NextFloat01();
                biases[i] = biases[i] * val + partner.biases[i] * (1.0f - val);
            }
        }
    }

    /// <summary>
    /// Neural network layer connection.
    /// </summary>
    public class NeuralNetworkLayerConnection
    {

        /// <summary>
        /// Minimum weight value when randomly generating weights.
        /// </summary>
        public static float MIN_WEIGHT = 0.0f;
        /// <summary>
        /// Maximum weight value when randomly generating weights.
        /// </summary>
        public static float MAX_WEIGHT = 0.06f;

        /// <summary>
        /// Number of synapses(neuron connections).
        /// </summary>
        public int numberOfSynapses;
        /// <summary>
        /// Array of synapse weights.
        /// </summary>
        public float[] weights;

        /// <summary>
        /// Create new struct from existing.
        /// </summary>
        /// <param name="src">Existing to copy.</param>
        public NeuralNetworkLayerConnection(NeuralNetworkLayerConnection src)
        {
            numberOfSynapses = src.numberOfSynapses;
            weights = new float[numberOfSynapses];
        }
        /// <summary>
        /// Create new struct.
        /// </summary>
        /// <param name="inLayer">Input layer.</param>
        /// <param name="outLayer">Output layer.</param>
        public NeuralNetworkLayerConnection(NeuralNetworkLayer inLayer, NeuralNetworkLayer outLayer)
        {
            numberOfSynapses = inLayer.numberOfNeurons * outLayer.numberOfNeurons;

            weights = new float[numberOfSynapses];
        }

        /// <summary>
        /// Randomly generate weights.
        /// </summary>
        public void RandomizeWeights()
        {
            int i = numberOfSynapses;
            while (i-- > 0)
            {
                weights[i] = Utils.NextFloat01() * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT;
            }
        }

        /// <summary>
        /// Copy weights from another connection struct.
        /// </summary>
        /// <param name="nnc">Source to copy from.</param>
        public void CopyWeights(NeuralNetworkLayerConnection nnc)
        {
            float[] cb = nnc.weights;

            int i = numberOfSynapses;
            while (i-- > 0)
            {
                weights[i] = cb[i];
            }
        }


        /// <summary>
        /// Mutates a selection of weights randomly.
        /// </summary>
        /// <param name="selectionChance">The chance(0-1) of a weight being mutated.</param>
        public void Mutate(float selectionChance)
        {
            int i = numberOfSynapses;
            while (i-- > 0)
            {
                if (Utils.NextFloat01() <= selectionChance)
                {
                    weights[i] = Utils.NextFloat01() * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT;
                }
            }
        }

        //breed with another layer connection data class(partner), partner must have the same # of synapses
        //takes a random selection of weights and biases from partner and replaces the a %(partPartner) of this classes weights/classes with those
        //partPartner is the % of weights and biases to use from the partner, 0 being none and 1 being all the weights/biases
        /// <summary>
        /// Breed with another connection(partner) taking a %(partPartner) of randomly selected weights.
        /// </summary>
        /// <param name="partner">Partner layer.</param>
        /// <param name="partPartner">Percent(0-1) of weights to take from partner.</param>
        public void Breed(NeuralNetworkLayerConnection partner)
        {
            int i = numberOfSynapses;
            while (i-- > 0)
            {
                //randomly mix
                float val = Utils.NextFloat01();
                weights[i] = weights[i] * val + partner.weights[i] * (1.0f - val);
            }
        }


        /// <summary>
        /// Save connection to stream.
        /// </summary>
        /// <param name="s">Stream.</param>
        public void Save(Stream s)
        {
            int i = numberOfSynapses;
            while (i-- > 0)
            {
                s.Write(Utils.FloatToBytes(weights[i]), 0, 4);
            }
        }

        //load data from stream
        /// <summary>
        /// Load connection from stream.
        /// </summary>
        /// <param name="s">Stream.</param>
        public void Load(Stream s)
        {
            byte[] buf = new byte[4];

            int i = numberOfSynapses;
            while (i-- > 0)
            {
                s.Read(buf, 0, 4);
                weights[i] = Utils.FloatFromBytes(buf);
            }
        }
    }




    

    //structure for saving network run-time state memory
    /// <summary>
    /// Neural network execution memory.
    /// </summary>
    public class NeuralNetworkContext
    {
        /// <summary>
        /// Input data.
        /// </summary>
        public float[] inputData;
        /// <summary>
        /// Output data.
        /// </summary>
        public float[] outputData;
        /// <summary>
        /// Hidden state data.
        /// </summary>
        public float[] hiddenData;
        /// <summary>
        /// Hidden recurring state data.
        /// </summary>
        public float[][] hiddenRecurringData;

        /// <summary>
        /// Allocate memory arrays.
        /// </summary>
        /// <param name="nn">Source network.</param>
        public void Setup(NeuralNetwork nn)
        {
            nn.SetupExecutionArrays(out inputData, out outputData, out hiddenData, out hiddenRecurringData);
            Reset(true);
        }

        /// <summary>
        /// Reset memory arrays.
        /// </summary>
        /// <param name="resetio">Should reset in/out arrays too?</param>
        public void Reset(bool resetio)
        {
            if (resetio)
            {
                Utils.Fill(outputData, 0.0f);
                Utils.Fill(inputData, 0.0f);
            }
            Utils.Fill(hiddenData, 0.0f);
            for (int i = 0; i < hiddenRecurringData.Length; i++)
            {
                if (hiddenRecurringData[i] != null) Utils.Fill(hiddenRecurringData[i], 0.0f);
            }
        }
    }

    /// <summary>
    /// Full execution memory(needed for training).
    /// </summary>
    public class NeuralNetworkFullContext
    {
        public float[][] hiddenBuffer, hiddenRecurringBuffer;

        public void Setup(NeuralNetwork nn)
        {
            hiddenBuffer = new float[nn.hiddenLayers.Length][];
            hiddenRecurringBuffer = new float[nn.hiddenLayers.Length][];
            for (int i = 0; i < nn.hiddenLayers.Length; i++)
            {
                hiddenBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                if (nn.hiddenLayers[i].recurring)
                {
                    hiddenRecurringBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                }
            }
        }
    }
}
