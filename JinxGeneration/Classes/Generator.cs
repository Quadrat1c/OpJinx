using System;
using System.Threading;

namespace JinxNeuralNetwork
{

    /// <summary>
    /// Used for generating input deltas from a desired target output.
    /// </summary>
    public class NeuralNetworkGenerator
    {

        private bool hasRecurring;

        private NeuralNetwork neuralNetwork;

        private NeuralNetworkContext[] stackedRuntimeContext;
        private NeuralNetworkFullContext[] stackedFullContext;
        private NeuralNetworkPropagationState[] stackedDerivativeMemory;
        private NeuralNetworkDerivativeMemory derivatives = new NeuralNetworkDerivativeMemory();
        private float[][] recurringBPBuffer;

        private int maxUnrollLength;

        /// <summary>
        /// Create new NeuralNetworkGenerator.
        /// </summary>
        /// <param name="nn">NeuralNetwork to train.</param>
        public NeuralNetworkGenerator(NeuralNetwork nn, int maxUnrollLen)
        {
            neuralNetwork = nn;
            maxUnrollLength = maxUnrollLen;

            //check for recurring layer, if need to stack and unroll
                for (int i = 0; i < nn.hiddenLayers.Length; i++)
                {
                    if (nn.hiddenLayers[i].recurring)
                    {
                        hasRecurring = true;
                        break;
                    }
                }
            

            derivatives.Setup(nn);

            if (hasRecurring)
            {
                recurringBPBuffer = new float[nn.hiddenLayers.Length][];
                for (int i = 0; i < recurringBPBuffer.Length - 1; i++)
                {
                    if (nn.hiddenLayers[i].recurring) recurringBPBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                }
            }
            else
            {
                maxUnrollLen = 1;
            }

                stackedRuntimeContext = new NeuralNetworkContext[maxUnrollLen];
                stackedFullContext = new NeuralNetworkFullContext[maxUnrollLen];
                stackedDerivativeMemory = new NeuralNetworkPropagationState[maxUnrollLen];
                for (int i = 0; i < maxUnrollLen; i++)
                {
                    stackedRuntimeContext[i] = new NeuralNetworkContext();
                    stackedRuntimeContext[i].Setup(nn);

                    stackedFullContext[i] = new NeuralNetworkFullContext();
                    stackedFullContext[i].Setup(nn);

                    stackedDerivativeMemory[i] = new NeuralNetworkPropagationState();
                    stackedDerivativeMemory[i].Setup(nn, stackedRuntimeContext[i], stackedFullContext[i], derivatives);
                    stackedDerivativeMemory[i].inputMem = new float[nn.inputLayer.numberOfNeurons];
                }
           
        }


        /// <summary>
        /// Generate input deltas for a recurring network.
        /// </summary>
        /// <param name="inputData"></param>
        /// <param name="targetData"></param>
        /// <param name="crossEntropy"></param>
        /// <returns>Array of input deltas.</returns>
        public float[][] InputErrorPropagationRecurring(float[][] inputData, float[][] targetData)
        {
            if (inputData.Length > maxUnrollLength)
            {
                throw new System.ArgumentException("Input/target array cannot be larger then max unroll length!");
            }
            if (!hasRecurring)
            {
                throw new System.ArgumentException("No recurring layers to perform recurring error propagation on.");
            }

            derivatives.Reset();

            for (int i = 0; i < inputData.Length; i++)
            {
                stackedRuntimeContext[i].Reset(true);
                stackedDerivativeMemory[i].Reset();
                Utils.Fill(stackedDerivativeMemory[i].inputMem, 0.0f);
            }

            //run forwardsand then run backwards backpropagating through recurring
            int dataIndex;
            for (dataIndex = 0; dataIndex < inputData.Length; dataIndex++)
            {
                Array.Copy(inputData[dataIndex], stackedRuntimeContext[dataIndex].inputData, stackedRuntimeContext[dataIndex].inputData.Length);
                neuralNetwork.Execute_FullContext(stackedRuntimeContext[dataIndex], stackedFullContext[dataIndex]);
                //neuralNetwork.ExecuteBackwards(targetData[dataArrayIndex][dataIndex], runtimeContext, fullContext, derivativeMemory);
            }
            //back propagate through stacked
            while (dataIndex-- > 0)
            {
                neuralNetwork.ExecuteBackwards(targetData[dataIndex], stackedRuntimeContext[dataIndex], stackedFullContext[dataIndex], stackedDerivativeMemory[dataIndex], 0, -1);
            }

            float[][] io = new float[inputData.Length][];
            for (int i = 0; i < io.Length; i++)
            {
                io[i] = new float[inputData[i].Length];
                Array.Copy(stackedDerivativeMemory[i].inputMem, io[i], io[i].Length);
            }
            return io;
        }

        /// <summary>
        /// Generate input deltas.
        /// </summary>
        /// <param name="inputData"></param>
        /// <param name="targetData"></param>
        /// <returns>Array of input deltas.</returns>
        public float[] InputErrorPropagation(float[] inputData, float[] targetData)
        {
            stackedRuntimeContext[0].Reset(false);
            stackedDerivativeMemory[0].Reset();
            Utils.Fill(stackedDerivativeMemory[0].inputMem, 0.0f);

            //simply run forwards and backwards
            Array.Copy(inputData, stackedRuntimeContext[0].inputData, stackedRuntimeContext[0].inputData.Length);
            neuralNetwork.Execute_FullContext(stackedRuntimeContext[0], stackedFullContext[0]);
            neuralNetwork.ExecuteBackwards(targetData, stackedRuntimeContext[0], stackedFullContext[0], stackedDerivativeMemory[0], NeuralNetworkTrainer.LOSS_TYPE_AVERAGE, -1);

            float[] io = new float[inputData.Length];
            Array.Copy(stackedDerivativeMemory[0].inputMem, io, io.Length);
            return io;
        }
    }
}
