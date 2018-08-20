namespace JinxNeuralNetwork
{
    /// <summary>
    /// NeuralNetwork executor.
    /// </summary>
    public class NeuralNetworkProgram
    {
        /// <summary>
        /// NeuralNetwork.
        /// </summary>
        public NeuralNetwork neuralNetwork;

        /// <summary>
        /// Execution memory like input/output.
        /// </summary>
        public NeuralNetworkContext context = new NeuralNetworkContext();

        /// <summary>
        /// Flag indicating new input data.
        /// </summary>
        public bool hasInput;
        /// <summary>
        /// Flag indicating new output data.
        /// </summary>
        public bool hasOutput;


        public int state = -1, total = -1;
        public float loss = 0.0f;

        /// <summary>
        /// Create new NeuralNetworkProgram for NeuralNetwork.
        /// </summary>
        /// <param name="nn"></param>
        public NeuralNetworkProgram(NeuralNetwork nn)
        {
            neuralNetwork = nn;

            context.Setup(nn);

            hasInput = false;
            hasOutput = false;
        }

        /// <summary>
        /// Execute NeuralNetwork program.
        /// </summary>
        public void Execute()
        {
            neuralNetwork.Execute(context);
            hasInput = false;
            hasOutput = true;
        }
    }
}
