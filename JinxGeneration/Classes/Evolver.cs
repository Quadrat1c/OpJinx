using System;
using System.Threading;
using System.IO;

namespace JinxNeuralNetwork
{

    /// <summary>
    /// Evolves a NeuralNetwork either through bruteforce evolution or breeding(genetic algorithm).
    /// </summary>
    public class NeuralNetworkEvolver
    {

        /// <summary>
        /// Stream next set of training data.
        /// </summary>
        public NeuralNetworkTrainer.StreamNextData onStreamNextData = null;

        /// <summary>
        /// Function callback for when no loss(0) has been reached.
        /// </summary>
        public ReachedGoalEventFunction onReachedGoal = null;
        /// <summary>
        /// Max loss before breeding is used.
        /// </summary>
        public float maxBreedingLoss = 1.0f;

        /// <summary>
        /// Desired loss goal.
        /// </summary>
        public float desiredLoss = 0.0f;

        /// <summary>
        /// Max mutation rate(0-1).
        /// </summary>
        public float maxMutationRate = 1.0f;

        public float minMutationRate = 0.0f;

        /// <summary>
        /// Rate that mutation rate increases.
        /// </summary>
        public float mutationIncreaseRate = 1e-3f;

        /// <summary>
        /// Sleep delay(ms).
        /// </summary>
        public int evolverDelay = 0;

        /// <summary>
        /// How to calculate loss.
        /// </summary>
        public int lossType = NeuralNetworkTrainer.LOSS_TYPE_AVERAGE;

        private NeuralNetwork sourceNetwork;
        private int numThreads;

        private bool breeding, first = true;

        private long generations = 0;

        private ProcessOutputInputGetLoss inputOutputLossFunction;
        private float[][] inputData, targetData;

        private float bestLoss = 1.0f, secondBestLoss = 1.0f, lossDelta = 0.0f;
        private NeuralNetwork bestNetwork, secondBestNetwork;

        private readonly object breedLock = new object();

        private Thread[] threads;
        private NeuralNetworkProgram[] subjects;
        private bool[] readyNextData;

        private bool running = true;


        /// <summary>
        /// Create new NeuralNetworkEvolver to train on specific input/target data.
        /// </summary>
        /// <param name="breed">Breeding flag.</param>
        /// <param name="sourcenn">NeuralNetwork to evolve.</param>
        /// <param name="nThreads">Number of threads to simulate evolution on.</param>
        /// <param name="inData">Input data.</param>
        /// <param name="targetDat">Target output data.</param>
        public NeuralNetworkEvolver(bool breed, NeuralNetwork sourcenn, int nThreads, float[][] inData, float[][] targetDat)
        {
            numThreads = nThreads;
            sourceNetwork = sourcenn;
            inputOutputLossFunction = premadePerfFunc;
            inputData = inData;
            targetData = targetDat;

            breeding = breed;

            threads = new Thread[numThreads];
            subjects = new NeuralNetworkProgram[numThreads];
            readyNextData = new bool[numThreads];

            for (int i = 0; i < nThreads; i++)
            {
                int ci = i;
                threads[i] = new Thread(() => evolverThread(ci));
                readyNextData[i] = false;

                subjects[i] = new NeuralNetworkProgram(sourcenn);
                subjects[i].neuralNetwork = new NeuralNetwork(sourceNetwork);
                subjects[i].neuralNetwork.CopyWeightsAndBiases(sourceNetwork);
            }
        }
        /// <summary>
        /// Create new NeuralNetworkEvolver to train using a custom performance function.
        /// </summary>
        /// <param name="breed">Breeding flag.</param>
        /// <param name="sourcenn">NeuralNetwork to evolve.</param>
        /// <param name="nThreads">Number of threads to simulate evolution on.</param>
        /// <param name="perfFunc">Performance/InputOutput processing function.</param>
        public NeuralNetworkEvolver(bool breed, NeuralNetwork sourcenn, int nThreads, ProcessOutputInputGetLoss lossFunc)
        {
            numThreads = nThreads;
            sourceNetwork = sourcenn;
            inputOutputLossFunction = lossFunc;

            breeding = breed;

            threads = new Thread[numThreads];
            subjects = new NeuralNetworkProgram[numThreads];
            readyNextData = new bool[numThreads];

            for (int i = 0; i < nThreads; i++)
            {
                int ci = i;
                threads[i] = new Thread(() => evolverThread(ci));
                readyNextData[i] = false;

                subjects[i] = new NeuralNetworkProgram(sourcenn);
                subjects[i].neuralNetwork = new NeuralNetwork(sourceNetwork);
                subjects[i].neuralNetwork.CopyWeightsAndBiases(sourceNetwork);
            }
        }

        /// <summary>
        /// Record NeuralNetworks loss.
        /// </summary>
        /// <param name="nn"></param>
        /// <param name="loss"></param>
        public void Record(NeuralNetwork nn, float loss)
        {
            if (loss <= 1.0f)
            {
                lock (breedLock)
                {
                    //returned performance, record performance and mutate new network
                    lossDelta += mutationIncreaseRate;
                    if (lossDelta > maxMutationRate)
                    {
                        lossDelta = maxMutationRate;
                    }
                    if (loss < bestLoss)
                    {
                        //new best loss
                        lossDelta = minMutationRate;
                        if (breeding)
                        {
                            secondBestLoss = bestLoss;
                            secondBestNetwork = bestNetwork;
                        }
                        bestLoss = loss;
                        bestNetwork = nn;
                    }
                    else if (breeding && loss < secondBestLoss)
                    {
                        secondBestLoss = loss;
                        secondBestNetwork = nn;
                    }
                }
            }

            generations++;
        }

        /// <summary>
        /// Create next generation.
        /// </summary>
        /// <returns>Next generation NeuralNetwork.</returns>
        public NeuralNetwork NextGeneration()
        {
            NeuralNetwork rtnn = new NeuralNetwork(sourceNetwork);

            //create new
            if (first)
            {
                rtnn.CopyWeightsAndBiases(sourceNetwork);
                first = false;
            }
            else
            {
                if (breeding && bestLoss <= maxBreedingLoss)
                {
                    if (bestNetwork != null && secondBestNetwork != null)
                    {
                        rtnn.CopyWeightsAndBiases(bestNetwork);
                        rtnn.Breed(secondBestNetwork);
                        if (lossDelta > 0.0f)
                        {
                            rtnn.Mutate(lossDelta);
                        }
                    }
                    else
                    {
                        rtnn.RandomizeWeightsAndBiases();
                    }
                }
                else
                {
                    rtnn.RandomizeWeightsAndBiases();
                }
            }

            return rtnn;
        }

        /// <summary>
        /// Record NeuralNetworks loss and create next generation.
        /// </summary>
        /// <param name="nn"></param>
        /// <param name="loss"></param>
        /// <returns>Next generation NeuralNetwork.</returns>
        public NeuralNetwork NextGeneration(NeuralNetwork nn, float loss)
        {
            NeuralNetwork rtnn = nn;

            if (first)
            {
                rtnn.CopyWeightsAndBiases(sourceNetwork);
                first = false;
            }
            else
            {
                if (loss <= 1.0f)
                {
                    lock (breedLock)
                    {
                        //returned performance, record performance and mutate new network
                        lossDelta += mutationIncreaseRate;
                        if (lossDelta > maxMutationRate)
                        {
                            lossDelta = maxMutationRate;
                        }
                        if (loss < bestLoss)
                        {
                            //new best loss
                            lossDelta = minMutationRate;
                            if (breeding)
                            {
                                secondBestLoss = bestLoss;
                                secondBestNetwork = bestNetwork;
                            }
                            bestLoss = loss;
                            bestNetwork = nn;
                            rtnn = new NeuralNetwork(nn);
                        }
                        else if (breeding && loss < secondBestLoss)
                        {
                            secondBestLoss = loss;
                            secondBestNetwork = nn;

                            rtnn = new NeuralNetwork(nn);
                        }


                        //create new
                        if (breeding && bestLoss <= maxBreedingLoss)
                        {
                            if (bestNetwork != null && secondBestNetwork != null)
                            {
                                rtnn.CopyWeightsAndBiases(bestNetwork);
                                rtnn.Breed(secondBestNetwork);
                                if (lossDelta > 0.0f)
                                {
                                    rtnn.Mutate(Utils.NextFloat01() * lossDelta);
                                }
                            }
                            else
                            {
                                rtnn.RandomizeWeightsAndBiases();
                            }
                        }
                        else
                        {
                            rtnn.RandomizeWeightsAndBiases();
                        }
                    }
                }
                else
                {
                    rtnn.RandomizeWeightsAndBiases();
                }
            }

            generations++;

            return rtnn;
        }


        private void evolverThread(int id)
        {
            NeuralNetworkProgram subject = subjects[id];

            while (running)
            {
                float loss = -1.0f;
                if (subject.state == -1)
                {
                    bool reset = true;
                    if (onStreamNextData != null)
                    {
                        readyNextData[id] = true;

                        if (id == 0)
                        {
                            while (true)
                            {
                                bool ready = true;
                                for (int i = 0; i < numThreads; i++)
                                {
                                    ready &= readyNextData[i];
                                }
                                if (ready) break;

                                Thread.Sleep(1);
                            }

                            reset = onStreamNextData(ref inputData, ref targetData);
                            for (int i = 0; i < numThreads; i++)
                            {
                                readyNextData[i] = false;
                            }
                        }
                        else
                        {
                            while (readyNextData[id]) Thread.Sleep(1);
                        }
                    }
                    if (reset)
                    {
                        if (subject.total > 0)
                        {
                            loss = subject.loss;
                            if (lossType == NeuralNetworkTrainer.LOSS_TYPE_AVERAGE) loss /= (float)subject.total;
                        }
                        subject.loss = 0.0f;
                    }
                }
                if (loss > -1.0f)
                {
                    subject.neuralNetwork = NextGeneration(subject.neuralNetwork, loss);

                    //reset state
                    subject.context.Reset(false);

                    if (loss <= desiredLoss)
                    {
                        //hit performance goal, done!
                        if (onReachedGoal != null) onReachedGoal();

                        //clean up
                        running = false;
                        return;
                    }
                }

                inputOutputLossFunction(subject);
                subject.Execute();

                if (evolverDelay != 0) Thread.Sleep(evolverDelay);
            }
        }


        private void premadePerfFunc(NeuralNetworkProgram nnp)
        {
            nnp.hasOutput = false;

            if (nnp.state != -1)
            {
                //calculate loss
                float perf = 0.0f;
                float[] odat = nnp.context.outputData,
                        tdat = targetData[nnp.state];
                int i = odat.Length;
                while (i-- > 0)
                {
                    float amax = Math.Abs(odat[i] - tdat[i]);
                    if (lossType == NeuralNetworkTrainer.LOSS_TYPE_AVERAGE)
                    {
                        perf += amax;
                    }
                    else
                    {
                        if (amax > perf) perf = amax;
                    }
                }
                if (lossType == NeuralNetworkTrainer.LOSS_TYPE_AVERAGE)
                {
                    nnp.loss += perf / (float)odat.Length;
                }
                else
                {
                    if (perf > nnp.loss) nnp.loss = perf;
                }

                //advance state
                nnp.state++;
                if (nnp.state >= targetData.Length)
                {
                    nnp.total += nnp.state;
                    nnp.state = -1;
                }
            }
            else
            {
                nnp.state = 0;
                nnp.total = 0;
            }

            if (nnp.state != -1)
            {
                //put next input data
                Array.Copy(inputData[nnp.state], nnp.context.inputData, inputData[nnp.state].Length);
                nnp.hasInput = true;
            }
        }

        /// <summary>
        /// Returns best neural network loss.
        /// </summary>
        /// <returns>Get loss of best generation.</returns>
        public float GetLoss()
        {
            return bestLoss;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns>Best performing generation NeuralNetwork.</returns>
        public NeuralNetwork GetBestNeuralNetwork()
        {
            return bestNetwork;
        }
        /// <summary>
        /// Returns number of generations tested.
        /// </summary>
        /// <returns>Number of generations tested.</returns>
        public long GetGenerations()
        {
            return generations;
        }

        /// <summary>
        /// Returns whether or not evolver has a best and second best network to do breeding with.
        /// </summary>
        /// <returns></returns>
        public bool HasBestAndSecondBest()
        {
            return (bestNetwork != null && secondBestNetwork != null);
        }


        public delegate void ProcessOutputInputGetLoss(NeuralNetworkProgram nnp);
        public delegate void ReachedGoalEventFunction();

        /// <summary>
        /// Stop simulating and wait for threads to stop(for infinite).
        /// </summary>
        public void Stop()
        {
            running = false;
            for (int i = 0; i < threads.Length; i++)
            {
                threads[i].Join(-1);
            }
        }

        /// <summary>
        /// Start simulating.
        /// </summary>
        public void Start()
        {
            running = true;
            for (int i = 0; i < threads.Length; i++)
            {
                threads[i].Start();
            }
        }

        /// <summary>
        /// Reset evolver state.
        /// </summary>
        public void Reset()
        {
            lossDelta = minMutationRate;
            bestLoss = 1.0f;
            secondBestLoss = 1.0f;
        }

        /// <summary>
        /// Save evolver state to stream.
        /// </summary>
        /// <param name="s"></param>
        public void Save(Stream s)
        {
            if (bestNetwork == null || secondBestNetwork == null) return;

            Utils.IntToStream((int)(generations / 10000000), s);
            Utils.FloatToStream(bestLoss, s);
            Utils.FloatToStream(secondBestLoss, s);
            bestNetwork.Save(s);
            secondBestNetwork.Save(s);
        }

        /// <summary>
        /// Load evolver state from stream.
        /// </summary>
        /// <param name="s"></param>
        public void Load(Stream s)
        {
            generations = Utils.IntFromStream(s) * 10000000;
            bestLoss = Utils.FloatFromStream(s);
            secondBestLoss = Utils.FloatFromStream(s);
            if (bestNetwork == null) bestNetwork = new NeuralNetwork(sourceNetwork);
            bestNetwork.Load(s);
            if (secondBestNetwork == null) secondBestNetwork = new NeuralNetwork(sourceNetwork);
            secondBestNetwork.Load(s);
        }


    }
}
