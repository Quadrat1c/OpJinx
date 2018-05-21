using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Epoch.Network;


namespace Epoch
{
    internal class Program
    {
        #region -- Constants --
        private const int MaxEpochs = 5000;
        private const double MinimumError = 0.01;
        private const TrainingType TrainingType = Network.TrainingType.MinimumError;
        #endregion

        #region -- Variables --
        private static int _numInputParameters;
        private static int _numHiddenLayerNeurons;
        private static int _numOutputParameters;
        private static Network.Network _network;
        private static List<DataSet> _dataSets;
        #endregion

        private static void Main()
        {
            string command;
            bool exit = false;

            ConsoleHelpers.Intro();

            while (!exit)
            {
                command = Console.ReadLine();

                switch (command)
                {
                    case "help":
                        Greet();
                        break;

                    case "setup":
                        SetupNetwork();
                        break;

                    case "train":
                        TrainNetwork();
                        break;

                    case "verify":
                        VerifyTraining();
                        break;

                    case "test":
                        testCustomNN(0.0f, 0.1f);
                        testCustomNN(0.1f, 0.0f);
                        testCustomNN(0.1f, 0.1f);
                        testCustomNN(0.0f, 0.0f);
                        break;

                    case "exit":
                        Environment.Exit(0);
                        break;

                    default:
                        Console.WriteLine("'" + command + "' unknown command! Type: 'help' for a list of commands.");
                        break;
                }
            }
        }

        #region Custom Neural Network
        private static void testCustomNN(float input1, float input2)
        {
            float hBias = -10;
            float oBias = 30;
            float h1 = 20 * input1 + 20 * input1 - hBias;
            float h2 = 20 * input2 + 20 * input2 - hBias;
            float output = 20 * h1 + 20 * h2 - oBias;
            Console.WriteLine(output);
        }

        #endregion

        #region -- Network Training --
        private static void TrainNetwork()
        {
            ConsoleHelpers.PrintNewLine();
            ConsoleHelpers.PrintUnderline(50);
            Console.WriteLine("Training...");

            Train();

            ConsoleHelpers.PrintNewLine();
            Console.WriteLine("Training Complete!");
            ConsoleHelpers.PrintNewLine();
        }

        private static void VerifyTraining()
        {
            Console.WriteLine("Let's test it!");
            ConsoleHelpers.PrintNewLine();

            while (true)
            {
                ConsoleHelpers.PrintUnderline(50);
                var values = GetInputData($"Type {_numInputParameters} inputs: ");
                var results = _network.Compute(values);
                ConsoleHelpers.PrintNewLine();

                foreach (var result in results)
                {
                    Console.WriteLine($"Output: {result}");
                }

                ConsoleHelpers.PrintNewLine();

                var convertedResults = new double[results.Length];
                for (var i = 0; i < results.Length; i++) { convertedResults[i] = results[i] > 0.5 ? 1 : 0; }

                var message = $"Was the result supposed to be {string.Join(" ", convertedResults)}? (yes/no/exit)";
                if (!ConsoleHelpers.GetBool(message))
                {
                    var offendingDataSet = _dataSets.FirstOrDefault(x => x.Values.SequenceEqual(values) && x.Targets.SequenceEqual(convertedResults));
                    _dataSets.Remove(offendingDataSet);

                    var expectedResults = GetExpectedResult("What were the expected results?");
                    if (!_dataSets.Exists(x => x.Values.SequenceEqual(values) && x.Targets.SequenceEqual(expectedResults)))
                        _dataSets.Add(new DataSet(values, expectedResults));

                    ConsoleHelpers.PrintNewLine();
                    Console.WriteLine("Retraining Network...");
                    ConsoleHelpers.PrintNewLine();

                    Train();
                }
                else
                {
                    ConsoleHelpers.PrintNewLine();
                    Console.WriteLine("Neat!");
                    Console.WriteLine("Encouraging Network...");
                    ConsoleHelpers.PrintNewLine();

                    Train();
                }
            }
        }

        private static void Train()
        {
            _network.Train(_dataSets, TrainingType == TrainingType.Epoch ? MaxEpochs : MinimumError);
        }
        #endregion

        #region -- Network Setup --
        private static void Greet()
        {
            Console.WriteLine("We're going to create an artificial Neural Network!");
            Console.WriteLine("The network will use back propagation to train itself.");
            ConsoleHelpers.PrintUnderline(50);
            Console.WriteLine("help - shows commands");
            Console.WriteLine("setup - create a neural network");
            Console.WriteLine("train - trains a neural network");
            Console.WriteLine("verify - run the neural network");
            Console.WriteLine("test - testCustomNN??");
            ConsoleHelpers.PrintNewLine();
        }

        private static void SetupNetwork()
        {
            if (ConsoleHelpers.GetBool("Do you want to read from the space delimited data.txt file? (yes/no/exit)"))
            {
                SetupFromFile();
            }
            else
            {
                SetNumInputParameters();
                SetNumNeuronsInHiddenLayer();
                SetNumOutputParameters();
                GetTrainingData();
            }

            Console.WriteLine("Creating Network...");
            _network = new Network.Network(_numInputParameters, _numHiddenLayerNeurons, _numOutputParameters);
            ConsoleHelpers.PrintNewLine();
        }

        private static void SetNumInputParameters()
        {
            ConsoleHelpers.PrintNewLine();
            Console.WriteLine("How many input parameters will there be? (2 or more)");
            _numInputParameters = ConsoleHelpers.GetInput("Input Parameters: ", 2);
            ConsoleHelpers.PrintNewLine(2);
        }

        private static void SetNumNeuronsInHiddenLayer()
        {
            Console.WriteLine("How many neurons in the hidden layer? (2 or more)");
            _numHiddenLayerNeurons = ConsoleHelpers.GetInput("Neurons: ", 2);
            ConsoleHelpers.PrintNewLine(2);
        }

        private static void SetNumOutputParameters()
        {
            Console.WriteLine("How many output parameters will there be? (1 or more)");
            _numOutputParameters = ConsoleHelpers.GetInput("Output Parameters: ", 1);
            ConsoleHelpers.PrintNewLine(2);
        }

        private static void GetTrainingData()
        {
            ConsoleHelpers.PrintUnderline(50);
            Console.WriteLine("Now, we need some input data.");
            ConsoleHelpers.PrintNewLine();

            _dataSets = new List<DataSet>();
            for (var i = 0; i < 4; i++)
            {
                var values = GetInputData($"Data Set {i + 1}");
                var expectedResult = GetExpectedResult($"Expected Result for Data Set {i + 1}:");
                _dataSets.Add(new DataSet(values, expectedResult));
            }
        }

        private static double[] GetInputData(string message)
        {
            Console.WriteLine(message);
            var line = ConsoleHelpers.GetLine();

            while (line == null || line.Split(' ').Length != _numInputParameters)
            {
                Console.WriteLine($"{_numInputParameters} inputs are required. Ex: 0 1");
                ConsoleHelpers.PrintNewLine();
                Console.WriteLine(message);
                line = ConsoleHelpers.GetLine();
            }

            var values = new double[_numInputParameters];
            var lineNums = line.Split(' ');
            for (var i = 0; i < lineNums.Length; i++)
            {
                double num;
                if (double.TryParse(lineNums[i], out num))
                {
                    values[i] = num;
                }
                else
                {
                    Console.WriteLine("You entered an invalid number.  Try again");
                    ConsoleHelpers.PrintNewLine(2);
                    return GetInputData(message);
                }
            }

            return values;
        }

        private static double[] GetExpectedResult(string message)
        {
            Console.WriteLine(message);
            var line = ConsoleHelpers.GetLine();

            while (line == null || line.Split(' ').Length != _numOutputParameters)
            {
                Console.WriteLine($"{_numOutputParameters} outputs are required.");
                ConsoleHelpers.PrintNewLine();
                Console.WriteLine(message);
                line = ConsoleHelpers.GetLine();
            }

            var values = new double[_numOutputParameters];
            var lineNums = line.Split(' ');
            for (var i = 0; i < lineNums.Length; i++)
            {
                int num;
                if (int.TryParse(lineNums[i], out num) && (num == 0 || num == 1))
                {
                    values[i] = num;
                }
                else
                {
                    Console.WriteLine("You must enter 1s and 0s!");
                    ConsoleHelpers.PrintNewLine(2);
                    return GetExpectedResult(message);
                }
            }

            return values;
        }
        #endregion

        #region -- I/O Help --
        private static void SetupFromFile()
        {
            _dataSets = new List<DataSet>();
            var fileContent = File.ReadAllText("data.txt");
            var lines = fileContent.Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries);

            if (lines.Length < 2)
            {
                ConsoleHelpers.WriteError("There aren't enough lines in the file.  The first line should have 3 integers representing the number of inputs, the number of hidden neurons and the number of outputs." +
                           "\r\nThere should also be at least one line of data.");
            }
            else
            {
                var setupParameters = lines[0].Split(' ');
                if (setupParameters.Length != 3)
                    ConsoleHelpers.WriteError("There aren't enough setup parameters.");

                if (!int.TryParse(setupParameters[0], out _numInputParameters) || !int.TryParse(setupParameters[1], out _numHiddenLayerNeurons) || !int.TryParse(setupParameters[2], out _numOutputParameters))
                    ConsoleHelpers.WriteError("The setup parameters are malformed.  There must be 3 integers.");

                if (_numInputParameters < 2)
                    ConsoleHelpers.WriteError("The number of input parameters must be greater than or equal to 2.");

                if (_numHiddenLayerNeurons < 2)
                    ConsoleHelpers.WriteError("The number of hidden neurons must be greater than or equal to 2.");

                if (_numOutputParameters < 1)
                    ConsoleHelpers.WriteError("The number of hidden neurons must be greater than or equal to 1.");
            }

            for (var lineIndex = 1; lineIndex < lines.Length; lineIndex++)
            {
                var items = lines[lineIndex].Split(' ');
                if (items.Length != _numInputParameters + _numOutputParameters)
                    ConsoleHelpers.WriteError($"The data file is malformed.  There were {items.Length} elements on line {lineIndex + 1} instead of {_numInputParameters + _numOutputParameters}");

                var values = new double[_numInputParameters];
                for (var i = 0; i < _numInputParameters; i++)
                {
                    double num;
                    if (!double.TryParse(items[i], out num))
                        ConsoleHelpers.WriteError($"The data file is malformed.  On line {lineIndex + 1}, input parameter {items[i]} is not a valid number.");
                    else
                        values[i] = num;
                }

                var expectedResults = new double[_numOutputParameters];
                for (var i = 0; i < _numOutputParameters; i++)
                {
                    int num;
                    if (!int.TryParse(items[_numInputParameters + i], out num))
                        Console.WriteLine($"The data file is malformed.  On line {lineIndex}, output paramater {items[i]} is not a valid number.");
                    else
                        expectedResults[i] = num;
                }
                _dataSets.Add(new DataSet(values, expectedResults));
            }
        }
        #endregion
    }
}
