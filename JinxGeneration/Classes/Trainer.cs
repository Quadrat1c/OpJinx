using System;
using System.Threading;
using System.IO;

namespace JinxNeuralNetwork
{

    /// <summary>
    /// Trains a NeuralNetwork on input and target data using AdaGrad.
    /// </summary>
    public class NeuralNetworkTrainer
    {
        public delegate bool StreamNextData(ref float[][] inp, ref float[][] targ);

        public const int LOSS_TYPE_AVERAGE = 0,
                         LOSS_TYPE_MAX = 1,
                         LOSS_TYPE_CROSSENTROPY = 2;

        public StreamNextData onStreamNextData = null;

        /// <summary>
        /// Delay(ms) in the execution thread.
        /// </summary>
        public int delay = 0;

        /// <summary>
        /// Learning rate(0-1).
        /// </summary>
        public float learningRate = 1e-1f;

        /// <summary>
        /// Enables stochastic skipping, skipping a random amount of training states to ensure uniformity.
        /// </summary>
        public bool stochasticSkipping = false;

        /// <summary>
        /// Desired loss to stop training at.
        /// </summary>
        public float desiredLoss = 1e-2f;

        /// <summary>
        /// Callback for when desiredLoss is reached.
        /// </summary>
        public NeuralNetworkEvolver.ReachedGoalEventFunction onReachedGoal = null;

        /// <summary>
        /// Amount of smoothing to apply to smooth loss regularization.
        /// </summary>
        public float lossSmoothing = 0.001f;

        /// <summary>
        /// Randomly shuffles input/training data order.
        /// </summary>
        public float shuffleChance = 0.0f;

        private int lossType;

        private int[] crossEntropyLossTargets;
        private float[][] inputData, targetData;
        private bool running = true,
                     hasRecurring;

        private NeuralNetwork neuralNetwork;

        private NeuralNetworkContext[] stackedRuntimeContext;
        private NeuralNetworkFullContext[] stackedFullContext;
        private NeuralNetworkPropagationState[] stackedDerivativeMemory;
        private NeuralNetworkDerivativeMemory derivatives = new NeuralNetworkDerivativeMemory();
        
        /// <summary>
        /// AdaGrad memory, use this for saving/loading training state.
        /// </summary>
        public NeuralNetworkAdaGradMemory adagradMemory = new NeuralNetworkAdaGradMemory();

        private float lossDelta, bestLoss, newLoss, smoothLoss;

        private int skipN, dataIndex, unrollCount, lossSampleCount;
        private long iterations;

        private Thread thread;

        private int maxUnrollLength;

        private bool resetState;


        /// <summary>
        /// Create new NeuralNetworkTrainer.
        /// </summary>
        /// <param name="nn">NeuralNetwork to train.</param>
        /// <param name="inputDat">Input data.</param>
        /// <param name="targetDat">Target data.</param>
        /// <param name="maxUnrollLen">Memory state unroll times, for recurring layers.</param>
        /// <param name="losType">Loss calculation type, NeuralNetworkTrainer.LOSS_TYPE_AVERAGE/MAX/CROSSENTROPY.</param>
        public NeuralNetworkTrainer(NeuralNetwork nn, float[][] inputDat, float[][] targetDat, int maxUnrollLen, int losType)
        {
            neuralNetwork = nn;
            inputData = inputDat;
            targetData = targetDat;

            maxUnrollLength = maxUnrollLen;
            if (maxUnrollLength < 1) maxUnrollLength = 1;

            lossType = losType;

            //check for recurring layer, if need to stack and unroll
            if (nn.outputLayer.recurring)
            {
                hasRecurring = true;
            }
            else
            {
                for (int i = 0; i < nn.hiddenLayers.Length; i++)
                {
                    if (nn.hiddenLayers[i].recurring)
                    {
                        hasRecurring = true;
                        break;
                    }
                }
            }

            derivatives.Setup(nn);
            adagradMemory.Setup(nn);
            adagradMemory.Reset();

            int tunrollLen = maxUnrollLength;
            if (!hasRecurring)
            {
                tunrollLen = 1;
            }

            stackedRuntimeContext = new NeuralNetworkContext[tunrollLen];
            stackedFullContext = new NeuralNetworkFullContext[tunrollLen];
            stackedDerivativeMemory = new NeuralNetworkPropagationState[tunrollLen];
            for (int i = 0; i < stackedRuntimeContext.Length; i++)
            {
                stackedRuntimeContext[i] = new NeuralNetworkContext();
                stackedRuntimeContext[i].Setup(nn);

                stackedFullContext[i] = new NeuralNetworkFullContext();
                stackedFullContext[i].Setup(nn);

                stackedDerivativeMemory[i] = new NeuralNetworkPropagationState();
                stackedDerivativeMemory[i].Setup(nn, stackedRuntimeContext[i], stackedFullContext[i], derivatives);
            }
        }


        /// <summary>
        /// Initialize data states for training but don't start thread.
        /// </summary>
        public void StartInit()
        {
            //reset adagrad memory
            adagradMemory.learningRate = learningRate;

            if (lossType == LOSS_TYPE_CROSSENTROPY)
            {
                bestLoss = smoothLoss = (float)-Math.Log(1.0f / neuralNetwork.outputLayer.numberOfNeurons) * maxUnrollLength;//initial smooth loss
            }
            else
            {
                smoothLoss = 1.0f;
                bestLoss = 1.0f;
            }

            lossDelta = 1.0f;
            unrollCount = 0;
            skipN = 0;
            dataIndex = 0;
            iterations = 0;
            resetState = true;
            if (onStreamNextData != null)
            {
                onStreamNextData(ref inputData, ref targetData);
            }
            if (lossType == LOSS_TYPE_CROSSENTROPY)
            {
                crossEntropyLossTargets = new int[targetData.Length];
                for (int i = 0; i < targetData.Length; i++)
                {

                    int r = Utils.Largest(targetData[i], 0, targetData[i].Length);
                    if (targetData[i][r] > 0.0f) crossEntropyLossTargets[i] = r;
                    else crossEntropyLossTargets[i] = -1;

                }
            }

            running = true;

        }



        /// <summary>
        /// Initializes data states and starts training.
        /// </summary>
        public void Start()
        {

            StartInit();

            //start
            thread = new Thread(processingThread);
            thread.Start();
        }
        /// <summary>
        /// Stop training.
        /// </summary>
        public void Stop()
        {
            running = false;
        }

        /// <summary>
        /// Joins training thread.
        /// </summary>
        /// <param name="timeout"></param>
        /// <returns></returns>
        public bool Join(int timeout)
        {
            return thread.Join(timeout);
        }

        /// <summary>
        /// Get best loss.
        /// </summary>
        /// <returns>Loss.</returns>
        public float GetLoss()
        {
            return bestLoss;
        }

        /// <summary>
        /// Get smoothed loss.
        /// </summary>
        /// <returns></returns>
        public float GetSmoothLoss()
        {
            return smoothLoss;
        }

        /// <summary>
        /// Get number of iterations.
        /// </summary>
        /// <returns></returns>
        public long GetIterations()
        {
            return iterations;
        }

        //copies c1 recurring data to c2
        private void CopyRecurringState(NeuralNetworkContext c1, NeuralNetworkContext c2)
        {
            int i = c1.hiddenRecurringData.Length;
            while (i-- > 0)
            {
                if (c1.hiddenRecurringData[i] == null) continue;
                Array.Copy(c1.hiddenRecurringData[i], c2.hiddenRecurringData[i], c1.hiddenRecurringData[i].Length);
            }
        }

        /// <summary>
        /// Run single iteration of learning, either 1 forward or backward propagation.
        /// </summary>
        public void Learn()
        {
            if (!running) return;

            if (resetState)
            {
                resetState = false;
                newLoss = 0.0f;
                lossSampleCount = 0;

                derivatives.Reset();
                for (int i = 0; i < stackedRuntimeContext.Length; i++)
                {
                    stackedRuntimeContext[i].Reset(true);
                    stackedDerivativeMemory[i].Reset();
                }

                if (hasRecurring && stochasticSkipping)
                {
                    if (targetData.Length < maxUnrollLength)
                    {
                        skipN = 0;
                    }
                    else
                    {
                        skipN = (int)(Utils.NextInt(0, (targetData[dataIndex].Length % maxUnrollLength) + 1));
                    }
                }
            }

            //run forwards for maxUnrollLength and then run backwards for maxUnrollLength backpropagating through recurring
            if (skipN > 0)
            {
                //skip random # at beginning to apply a 'shuffle'
                if (hasRecurring)
                {
                    Array.Copy(inputData[dataIndex], stackedRuntimeContext[0].inputData, stackedRuntimeContext[0].inputData.Length);
                    neuralNetwork.Execute(stackedRuntimeContext[0]);
                }
                skipN--;
            }
            else
            {
                int unrollIndex = unrollCount;
                if (!hasRecurring) unrollIndex = 0;
                Array.Copy(inputData[dataIndex], stackedRuntimeContext[unrollIndex].inputData, stackedRuntimeContext[unrollIndex].inputData.Length);
                neuralNetwork.Execute_FullContext(stackedRuntimeContext[unrollIndex], stackedFullContext[unrollIndex]);

                unrollCount++;
                if (hasRecurring)
                {
                    if (unrollCount >= maxUnrollLength || dataIndex + 1 >= targetData.Length)
                    {
                        //back propagate through stacked
                        float nextLoss = 0.0f;
                        int tdatIndex = dataIndex,
                            nunroll = unrollCount;
                        while (unrollCount-- > 0)
                        {
                            neuralNetwork.ExecuteBackwards(targetData[tdatIndex], stackedRuntimeContext[unrollCount], stackedFullContext[unrollCount], stackedDerivativeMemory[unrollCount], lossType, (lossType == LOSS_TYPE_CROSSENTROPY ? crossEntropyLossTargets[tdatIndex] : -1));
                            if (lossType == LOSS_TYPE_AVERAGE)
                            {
                                nextLoss += stackedDerivativeMemory[unrollCount].loss;
                            }
                            else
                            {
                                if (stackedDerivativeMemory[unrollCount].loss > nextLoss)
                                {
                                    nextLoss = stackedDerivativeMemory[unrollCount].loss;
                                }
                            }

                            tdatIndex--;
                        }

                        if (lossType == LOSS_TYPE_AVERAGE)
                        {
                            newLoss += nextLoss/(float)nunroll;
                            lossSampleCount++;
                        }
                        else
                        {
                            if (nextLoss > newLoss) newLoss = nextLoss;
                        }
                        //learn
                        adagradMemory.Apply(stackedDerivativeMemory[0]);
                        derivatives.Reset();

                        unrollCount = 0;


                        //copy recurring state over
                        CopyRecurringState(stackedRuntimeContext[maxUnrollLength - 1], stackedRuntimeContext[0]);
                    }
                    else
                    {
                        //copy recurring state into next
                        CopyRecurringState(stackedRuntimeContext[unrollCount - 1], stackedRuntimeContext[unrollCount]);
                    }
                }
                else
                {
                    neuralNetwork.ExecuteBackwards(targetData[dataIndex], stackedRuntimeContext[unrollIndex], stackedFullContext[unrollIndex], stackedDerivativeMemory[unrollIndex], lossType, (lossType == LOSS_TYPE_CROSSENTROPY ? crossEntropyLossTargets[dataIndex] : -1));
                    if (lossType == LOSS_TYPE_AVERAGE)
                    {
                        newLoss += stackedDerivativeMemory[unrollIndex].loss;
                        lossSampleCount++;
                    }
                    else
                    {
                        if (stackedDerivativeMemory[unrollIndex].loss > newLoss)
                        {
                            newLoss = stackedDerivativeMemory[unrollIndex].loss;
                        }
                    }
                    if (unrollCount >= maxUnrollLength || dataIndex + 1 >= targetData.Length)
                    {
                        //learn
                        adagradMemory.Apply(stackedDerivativeMemory[0]);
                        derivatives.Reset();

                        unrollCount = 0;
                    }
                }
            }

            //advance index
            dataIndex++;
            if (dataIndex >= targetData.Length)
            {
                iterations++;
                dataIndex = 0;

                if (lossType == LOSS_TYPE_AVERAGE) newLoss /= (float)lossSampleCount;

                if (newLoss < bestLoss)
                {
                    bestLoss = newLoss;
                }
                if (newLoss <= desiredLoss)
                {
                    //hit goal, stop
                    if (onReachedGoal != null) onReachedGoal();
                    running = false;
                    return;
                }

                float lsl = smoothLoss;
                smoothLoss = smoothLoss * lossSmoothing + newLoss * (1.0f-lossSmoothing);
                lossDelta = lossDelta * lossSmoothing + (lsl - smoothLoss) * (1.0f-lossSmoothing);
                
                lossSampleCount = 0;
                newLoss = 0.0f;


                //stream new data
                if (onStreamNextData != null)
                {
                    resetState = onStreamNextData(ref inputData, ref targetData);
                    if (lossType == LOSS_TYPE_CROSSENTROPY)
                    {
                        crossEntropyLossTargets = new int[targetData.Length];
                        for (int i = 0; i < targetData.Length; i++)
                        {

                            int r = Utils.Largest(targetData[i], 0, targetData[i].Length);
                            if (targetData[i][r] > 0.0f) crossEntropyLossTargets[i] = r;
                            else crossEntropyLossTargets[i] = -1;

                        }
                    }
                }
                else
                {
                    resetState = true;
                }

                if (shuffleChance > 0.0f && Utils.NextFloat01() < shuffleChance)
                {
                    Utils.Shuffle(inputData, targetData);
                }
            }
        }

        private void processingThread()
        {

            while (running)
            {
                Learn();
                if (delay > 0) Thread.Sleep(delay);
            }

        }

        /// <summary>
        /// Returns true if StartInit/Start init has been called.
        /// </summary>
        /// <returns></returns>
        public bool Running()
        {
            return running;
        }

        /// <summary>
        /// Get loss delta.
        /// </summary>
        /// <returns></returns>
        public float GetLossDelta()
        {
            return lossDelta;
        }
    }



    /// <summary>
    /// Derivative memory.
    /// </summary>
    public class NeuralNetworkDerivativeMemory
    {
        public float[][] weightMems, biasMems, recurrWeightMems, outputFullConnectedWeightMems, recurringBPBuffer, altRecurringBPBuffer;

        public void Setup(NeuralNetwork nn)
        {
            biasMems = new float[nn.hiddenLayers.Length + 1][];
            weightMems = new float[nn.hiddenLayers.Length + 1][];
            recurrWeightMems = new float[nn.hiddenLayers.Length][];
            recurringBPBuffer = new float[nn.hiddenLayers.Length][];
            altRecurringBPBuffer = new float[nn.hiddenLayers.Length][];

            for (int i = 0; i < nn.hiddenLayers.Length; i++)
            {
                weightMems[i] = new float[nn.hiddenConnections[i].numberOfSynapses];
                biasMems[i] = new float[nn.hiddenLayers[i].numberOfNeurons];

                if (nn.hiddenLayers[i].recurring)
                {
                    recurrWeightMems[i] = new float[nn.hiddenRecurringConnections[i].numberOfSynapses];
                    recurringBPBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                    altRecurringBPBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                }
            }

            int lid = nn.hiddenLayers.Length;
            biasMems[lid] = new float[nn.outputLayer.numberOfNeurons];
            weightMems[lid] = new float[nn.outputConnection.numberOfSynapses];
        }

        public void SwapBPBuffers()
        {
            float[][] temp = recurringBPBuffer;
            recurringBPBuffer = altRecurringBPBuffer;
            altRecurringBPBuffer = temp;
        }

        public void Reset()
        {
            for (int i = 0; i < biasMems.Length; i++)
            {
                Utils.Fill(biasMems[i], 0.0f);
                Utils.Fill(weightMems[i], 0.0f);
                if (i < recurrWeightMems.Length && recurrWeightMems[i] != null)
                {
                    Utils.Fill(recurrWeightMems[i], 0.0f);
                    Utils.Fill(recurringBPBuffer[i], 0.0f);
                    Utils.Fill(altRecurringBPBuffer[i], 0.0f);
                }
                if (outputFullConnectedWeightMems != null && i < outputFullConnectedWeightMems.Length) Utils.Fill(outputFullConnectedWeightMems[i], 0.0f);
            }
        }

        public void Scale(float s)
        {
            for (int i = 0; i < biasMems.Length; i++)
            {
                Utils.Multiply(biasMems[i], s);
                Utils.Multiply(weightMems[i], s);
                if (i < recurrWeightMems.Length && recurrWeightMems[i] != null)
                {
                    Utils.Multiply(recurrWeightMems[i], s);
                }
                if (outputFullConnectedWeightMems != null && i < outputFullConnectedWeightMems.Length) Utils.Multiply(outputFullConnectedWeightMems[i], s);
            }
        }

        public void ResetOnlyBuffer()
        {
            for (int i = 0; i < biasMems.Length; i++)
            {
                if (i < recurrWeightMems.Length && recurrWeightMems[i] != null)
                {
                    Utils.Fill(recurringBPBuffer[i], 0.0f);
                    Utils.Fill(altRecurringBPBuffer[i], 0.0f);
                }
            }
        }
    }

    /// <summary>
    /// Back propagation state.
    /// </summary>
    public class NeuralNetworkPropagationState
    {
        public float loss;
        public float[][] weights, biases, recurrWeights,
                         weightMems, biasMems, recurrWeightMems,
                         buf, recurrBuf, state;
        public float[] inputMem;
        public NeuralNetworkDerivativeMemory derivativeMemory;

        public void Setup(NeuralNetwork nn, NeuralNetworkContext context, NeuralNetworkFullContext fullCtx, NeuralNetworkDerivativeMemory derivMem)
        {
            //initialize memory buffers
            state = new float[nn.hiddenLayers.Length][];
            weights = new float[nn.hiddenLayers.Length + 1][];
            biases = new float[nn.hiddenLayers.Length + 1][];

            buf = new float[nn.hiddenLayers.Length + 1][];
            recurrBuf = new float[nn.hiddenLayers.Length][];

            biasMems = derivMem.biasMems;
            weightMems = derivMem.weightMems;
            recurrWeightMems = derivMem.recurrWeightMems;
            recurrWeights = new float[nn.hiddenLayers.Length][];
            derivativeMemory = derivMem;

            for (int i = 0; i < nn.hiddenLayers.Length; i++)
            {
                state[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                weights[i] = nn.hiddenConnections[i].weights;
                biases[i] = nn.hiddenLayers[i].biases;

                if (i == 0)
                {
                    buf[i] = context.inputData;
                }
                else
                {
                    buf[i] = fullCtx.hiddenBuffer[i - 1];
                }


                if (nn.hiddenLayers[i].recurring)
                {
                    recurrWeights[i] = nn.hiddenRecurringConnections[i].weights;
                    recurrBuf[i] = fullCtx.hiddenRecurringBuffer[i];
                }
            }

            int lid = nn.hiddenLayers.Length;
            weights[lid] = nn.outputConnection.weights;
            biases[lid] = nn.outputLayer.biases;
            if (lid > 0)
            {
                buf[lid] = fullCtx.hiddenBuffer[lid - 1];
            }
            else
            {
                buf[lid] = context.inputData;
            }
        }


        public void Reset()
        {
            loss = 0.0f;
            for (int i = 0; i < buf.Length; i++)
            {
                Utils.Fill(buf[i], 0.0f);
                if (i < recurrBuf.Length && recurrBuf[i] != null) Utils.Fill(recurrBuf[i], 0.0f);
            }

        }
    }

    /// <summary>
    /// Struct for storing per-parameter learning rates when doing AdaGrad.
    /// </summary>
    public class NeuralNetworkAdaGradMemory
    {
        private const float EXPLODING_GRADIENT_CLAMP = 1.0f;
        private const double SQRT_EPSILON = 1e-8;

        public float learningRate;
        public float[][] weights, biases, recurringWeights;

        public void Setup(NeuralNetwork nn)
        {
            weights = new float[nn.hiddenLayers.Length + 1][];
            biases = new float[nn.hiddenLayers.Length + 1][];
            recurringWeights = new float[nn.hiddenLayers.Length][];

            for (int i = 0; i < nn.hiddenLayers.Length; i++)
            {
                weights[i] = new float[nn.hiddenConnections[i].numberOfSynapses];
                biases[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                if (nn.hiddenLayers[i].recurring) recurringWeights[i] = new float[nn.hiddenRecurringConnections[i].numberOfSynapses];
            }

            int lid = nn.hiddenLayers.Length;
            weights[lid] = new float[nn.outputConnection.numberOfSynapses];
            biases[lid] = new float[nn.outputLayer.numberOfNeurons];
        }

        /// <summary>
        /// Reset AdaGrad memory to 0.
        /// </summary>
        public void Reset()
        {
            for (int i = 0; i < weights.Length; i++)
            {
                Utils.Fill(weights[i], 0.0f);
                Utils.Fill(biases[i], 0.0f);
                if (i < recurringWeights.Length && recurringWeights[i] != null) Utils.Fill(recurringWeights[i], 0.0f);
            }
        }

        /// <summary>
        /// Partially reset adagrad memory, new memory = old memory * am.
        /// </summary>
        /// <param name="am"></param>
        public void ResetPartial(float am)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                Utils.Multiply(weights[i], am);
                Utils.Multiply(biases[i], am);

                if (i < recurringWeights.Length && recurringWeights[i] != null) Utils.Multiply(recurringWeights[i], am);
            }
        }

        /// <summary>
        /// Save AdaGrad memory to stream.
        /// </summary>
        /// <param name="s"></param>
        public void Save(Stream s)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                float[] f = weights[i];
                int k = f.Length;
                while (k-- > 0)
                {
                    s.Write(Utils.FloatToBytes(f[k]), 0, 4);
                }

                f = biases[i];
                k = f.Length;
                while (k-- > 0)
                {
                    s.Write(Utils.FloatToBytes(f[k]), 0, 4);
                }

                if (i < recurringWeights.Length && recurringWeights[i] != null)
                {
                    f = recurringWeights[i];
                    k = f.Length;
                    while (k-- > 0)
                    {
                        s.Write(Utils.FloatToBytes(f[k]), 0, 4);
                    }
                }
            }
        }
        /// <summary>
        /// Load AdaGrad memory from stream.
        /// </summary>
        /// <param name="s"></param>
        public void Load(Stream s)
        {
            byte[] b = new byte[4];
        
            for (int i = 0; i < weights.Length; i++)
            {
                float[] f = weights[i];
                int k = f.Length;
                while (k-- > 0)
                {
                    s.Read(b, 0, 4);
                    f[k] = Utils.FloatFromBytes(b);
                }

                f = biases[i];
                k = f.Length;
                while (k-- > 0)
                {
                    s.Read(b, 0, 4);
                    f[k] = Utils.FloatFromBytes(b);
                }

                if (i < recurringWeights.Length && recurringWeights[i] != null)
                {
                    f = recurringWeights[i];
                    k = f.Length;
                    while (k-- > 0)
                    {
                        s.Read(b, 0, 4);
                        f[k] = Utils.FloatFromBytes(b);
                    }
                }
            }
        }

        /// <summary>
        /// Add derivatives to learning rate and apply to network weights/biases
        /// </summary>
        /// <param name="derivMem"></param>
        public void Apply(NeuralNetworkPropagationState derivMem)
        {
            Apply(derivMem, derivMem.weights, derivMem.biases, derivMem.recurrWeights);
        }
        /// <summary>
        /// Add derivatives to learning rate and apply to network weights/biases.
        /// </summary>
        /// <param name="derivMem"></param>
        /// <param name="weight"></param>
        /// <param name="bias"></param>
        /// <param name="recurrWeight"></param>
        public void Apply(NeuralNetworkPropagationState derivMem, float[][] weight, float[][] bias, float[][] recurrWeight)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                float[] t = weights[i],
                        f = derivMem.weightMems[i],
                        w = weight[i];

                int k = f.Length;
                while (k-- > 0)
                {
                    float m = t[k],
                          d = f[k];

                    if (d < -EXPLODING_GRADIENT_CLAMP) d = -EXPLODING_GRADIENT_CLAMP;
                    else if (d > EXPLODING_GRADIENT_CLAMP) d = EXPLODING_GRADIENT_CLAMP;

                    m += d * d;
                    w[k] -= (learningRate * d) / (float)Math.Sqrt(m + SQRT_EPSILON);

                    t[k] = m;
                }




                t = biases[i];
                f = derivMem.biasMems[i];
                w = bias[i];

                k = f.Length;
                while (k-- > 0)
                {
                    float m = t[k],
                          d = f[k];

                    if (d < -EXPLODING_GRADIENT_CLAMP)
                    {
                        d = -EXPLODING_GRADIENT_CLAMP;
                    }
                    else
                    {
                        if (d > EXPLODING_GRADIENT_CLAMP)
                        {
                            d = EXPLODING_GRADIENT_CLAMP;
                        }
                    }

                    m += d * d;
                    w[k] -= (learningRate * d) / (float)Math.Sqrt(m + SQRT_EPSILON);

                    t[k] = m;
                }



                t = i < recurringWeights.Length ? recurringWeights[i] : null;
                if (t != null)
                {
                    f = derivMem.recurrWeightMems[i];
                    w = recurrWeight[i];

                    k = f.Length;
                    while (k-- > 0)
                    {
                        float m = t[k],
                              d = f[k];

                        if (d < -EXPLODING_GRADIENT_CLAMP)
                        {
                            d = -EXPLODING_GRADIENT_CLAMP;
                        }
                        else
                        {
                            if (d > EXPLODING_GRADIENT_CLAMP)
                            {
                                d = EXPLODING_GRADIENT_CLAMP;
                            }
                        }

                        m += d * d;
                        w[k] -= (learningRate * d) / (float)Math.Sqrt(m + SQRT_EPSILON);

                        t[k] = m;
                    }
                }


            }
        }

        /// <summary>
        /// Add derivatives without per-parameter learning rate.
        /// </summary>
        /// <param name="derivMem"></param>
        /// <param name="weight"></param>
        /// <param name="bias"></param>
        /// <param name="recurrWeight"></param>
        public static void ApplyNoMemory(NeuralNetworkPropagationState derivMem, float[][] weight, float[][] bias, float[][] recurrWeight, float learningRate)
        {
            for (int i = 0; i < weight.Length; i++)
            {
                float[] f = derivMem.weightMems[i],
                        w = weight[i];

                int k = f.Length;
                while (k-- > 0)
                {
                    float d = f[k];
                    if (d < -EXPLODING_GRADIENT_CLAMP) d = -EXPLODING_GRADIENT_CLAMP;
                    else if (d > EXPLODING_GRADIENT_CLAMP) d = EXPLODING_GRADIENT_CLAMP;

                    w[k] -= (learningRate * d);
                }


                f = derivMem.biasMems[i];
                w = bias[i];

                k = f.Length;
                while (k-- > 0)
                {
                    float d = f[k];

                    if (d < -EXPLODING_GRADIENT_CLAMP)
                    {
                        d = -EXPLODING_GRADIENT_CLAMP;
                    }
                    else
                    {
                        if (d > EXPLODING_GRADIENT_CLAMP)
                        {
                            d = EXPLODING_GRADIENT_CLAMP;
                        }
                    }

                    w[k] -= (learningRate * d);
                }



                if (recurrWeight[i] != null)
                {
                    f = derivMem.recurrWeightMems[i];
                    w = recurrWeight[i];

                    k = f.Length;
                    while (k-- > 0)
                    {
                        float d = f[k];

                        if (d < -EXPLODING_GRADIENT_CLAMP)
                        {
                            d = -EXPLODING_GRADIENT_CLAMP;
                        }
                        else
                        {
                            if (d > EXPLODING_GRADIENT_CLAMP)
                            {
                                d = EXPLODING_GRADIENT_CLAMP;
                            }
                        }

                        w[k] -= (learningRate * d);
                    }
                }
            }
        }
    }
}
