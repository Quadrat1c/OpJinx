using System.IO;
using System.Collections.Generic;

namespace JinxNeuralNetwork
{
    class Program
    {
        public static void Main(string[] args)
        {
            //read example text
            string exampleTxt = File.ReadAllText("example.txt");

            //encode in one-hot encoding, each letter is assigned a unique neuron.
            List<char> dict = null;
            float[][] data = Utils.EncodeStringOneHot(exampleTxt, out dict);

            //create neural network
            int hlen = exampleTxt.Length / 4;// dict.Count * 2;
            NeuralNetwork neuralNetwork = new NeuralNetwork(new NeuralNetworkLayer(dict.Count, false, null),//input layer
                                                            new NeuralNetworkLayer[] {//hidden layers
                                                                new NeuralNetworkLayer(hlen, true, Utils.Rectifier_ActivationFunction),
                                                            },
                                                            new NeuralNetworkLayer(dict.Count, false, Utils.Rectifier_ActivationFunction)//output layer
                                                            );
            neuralNetwork.RandomizeWeightsAndBiases(0.0f, 0.0f, 0.0f, 0.5f / hlen);


            //convert into training data for neural network

            //we want to input the current text character and have the neural network output the next text character.
            float[][] inputDat = new float[data.Length - 1][],
                     targetDat = new float[data.Length - 1][];
            for (int i = 0; i < data.Length - 1; i++)
            {
                inputDat[i] = data[i];
                targetDat[i] = data[i + 1];
            }


            //train neural network and every 10 seconds output text generated from the network
            NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(neuralNetwork, inputDat, targetDat, 2, NeuralNetworkTrainer.LOSS_TYPE_AVERAGE);
            trainer.learningRate = 0.04f;
            trainer.desiredLoss = 0.0f;
            trainer.lossSmoothing = 0.0f;

            //program for generating text
            NeuralNetworkProgram program = new NeuralNetworkProgram(neuralNetwork);

            //init trainer for training
            trainer.StartInit();

            long lastTime = 0,
                 nowTime;
            while (true)
            {
                trainer.Learn();

                nowTime = System.DateTime.Now.Ticks / 10000;
                if (nowTime - lastTime > 10000)
                {
                    //generate text(100-200 chars) from network
                    string txt = "";

                    program.context.Reset(true);
                    float[] id = program.context.inputData,
                            od = program.context.outputData;

                    int currentChar = Utils.NextInt(0, dict.Count),//random starting character
                        nChars = Utils.NextInt(100, 200);//between 100-200 characters
                    txt += dict[currentChar];
                    for (int i = 0; i < nChars; i++)
                    {
                        //input current character to network
                        id[currentChar] = 1.0f;

                        //run network
                        program.Execute();

                        //manually clear last character from input
                        id[currentChar] = 0.0f;


                        //treat output characters as probabilities, randomly sampling next character from probabilities
                        Utils.Normalize(od);
                        currentChar = Utils.RandomChoice(od);

                        txt += dict[currentChar];
                    }

                    //output iterations, loss and generated text
                    System.Console.WriteLine("Iterations: " + trainer.GetIterations() + ", Loss: " + trainer.GetLoss() + System.Environment.NewLine +
                                             "Text: " + txt + System.Environment.NewLine);

                    lastTime = nowTime;
                }
            }
        }
    }
}
