using System;
using System.IO;
using System.Net;
using System.Collections.Generic;

namespace JinxNeuralNetwork
{
    public class Utils {
        //a few util functions used 


        /// <summary>
        /// Compile NeuralNetwork into image-processing GLSL shader, only supports rectifier activation function and all layer sizes must be divisble by 3. Shader expects input in float3 array 'is'.
        /// </summary>
        /// <param name="nn"></param>
        /// <returns></returns>
        public static string AsShader(NeuralNetwork neuralNet)
        {
            int KERNEL_AREA = neuralNet.inputLayer.numberOfNeurons/3;

            //save neural network as glsl shader
            string shaderSrc = "";
            for (int h = 0; h < neuralNet.hiddenLayers.Length+1; h++)
            {
                NeuralNetworkLayer layer;
                NeuralNetworkLayerConnection connect;
                int isize;
                string ov,iv;
                if (h == 0)
                {
                    iv = "is";
                    isize = KERNEL_AREA;
                }
                else
                {
                    iv = "hs" + (h - 1);
                    isize = neuralNet.hiddenLayers[h - 1].numberOfNeurons / 3;
                }
                if (h >= neuralNet.hiddenLayers.Length)
                {
                    ov = "os";
                    layer = neuralNet.outputLayer;
                    connect = neuralNet.outputConnection;
                }
                else
                {
                    ov = "hs"+h;
                    layer = neuralNet.hiddenLayers[h];
                    connect = neuralNet.hiddenConnections[h];
                }

                int osize = layer.numberOfNeurons/3;
                shaderSrc += "vec3 "+ ov + "[" + osize + "];"+Environment.NewLine;

                int weightIndex = 0,
                    k = osize;
                while (k-- > 0)
                {
                    shaderSrc += ov + "[" + k + "] = clamp(vec3(";

                    int c = 3;
                    while (c-- > 0)
                    {
                        shaderSrc += layer.biases[(k * 3 + 2) - c].ToString(".0######") + "+";

                        int w = isize;
                        while (w-- > 0)
                        {
                            shaderSrc += "dot(" + iv + "[" + w + "],vec3(" +
                                connect.weights[weightIndex+2].ToString(".0######") + "," +
                                connect.weights[weightIndex + 1].ToString(".0######") + "," +
                                connect.weights[weightIndex].ToString(".0######") + "))";

                            if (w != 0) shaderSrc += "+";
                            weightIndex += 3;
                        }

                        if (c != 0) shaderSrc += ",";
                        else shaderSrc += "),0.,1.);" + Environment.NewLine;
                    }

                    if (k == 0) shaderSrc += Environment.NewLine;
                }
            }

            return shaderSrc;
        }

        /// <summary>
        /// Compile forward+backward propagating NeuralNetwork into image-processing GLSL shader, only supports rectifier activation function and all layer sizes must be divisble by 3. Shader expects input in float3 array 'is' and target output as 'ts'.
        /// </summary>
        /// <param name="neuralNet"></param>
        /// <returns></returns>
        public static string GenerationAsShader(NeuralNetwork neuralNet)
        {
            int KERNEL_AREA = neuralNet.inputLayer.numberOfNeurons / 3;

            //save neural network as glsl shader
            string shaderSrc = "//forward propagation"+Environment.NewLine;
            for (int h = 0; h < neuralNet.hiddenLayers.Length+1; h++)
            {
                NeuralNetworkLayer layer;
                NeuralNetworkLayerConnection connect;
                int isize;
                string ov, iv;
                if (h == 0)
                {
                    iv = "is";
                    isize = KERNEL_AREA;
                }
                else
                {
                    iv = "hs" + (h - 1);
                    isize = neuralNet.hiddenLayers[h - 1].numberOfNeurons / 3;
                }
                if (h >= neuralNet.hiddenLayers.Length)
                {
                    ov = "os";
                    layer = neuralNet.outputLayer;
                    connect = neuralNet.outputConnection;
                }
                else
                {
                    ov = "hs" + h;
                    layer = neuralNet.hiddenLayers[h];
                    connect = neuralNet.hiddenConnections[h];
                }

                int osize = layer.numberOfNeurons / 3;
                shaderSrc += "vec3 " + ov + "[" + osize + "];" + Environment.NewLine;

                int weightIndex = 0,
                    k = osize;
                while (k-- > 0)
                {
                    shaderSrc += ov + "[" + k + "] = clamp(vec3(";

                    int c = 3;
                    while (c-- > 0)
                    {
                        shaderSrc += layer.biases[(k * 3 + 2) - c].ToString(".0######") + "+";

                        int w = isize;
                        while (w-- > 0)
                        {
                            shaderSrc += "dot(" + iv + "[" + w + "],vec3(" +
                                connect.weights[weightIndex + 2].ToString(".0######") + "," +
                                connect.weights[weightIndex + 1].ToString(".0######") + "," +
                                connect.weights[weightIndex].ToString(".0######") + "))";

                            if (w != 0) shaderSrc += "+";
                            weightIndex += 3;
                        }

                        if (c != 0) shaderSrc += ",";
                        else shaderSrc += "),0.,1.);" + Environment.NewLine;
                    }

                    if (k == 0) shaderSrc += Environment.NewLine;
                }
            }

            int nout = neuralNet.outputLayer.numberOfNeurons / 3;
            shaderSrc += Environment.NewLine + "//target output difference/deriv" + Environment.NewLine +
                "vec3 td[" + nout + "];" + Environment.NewLine;
            for (int i = 0; i < nout; i++)
            {
                shaderSrc += "td[" + i + "] = os[" + i + "]-ts[" + i + "];" + Environment.NewLine;
            }

            shaderSrc += "//back propagation" + Environment.NewLine;
            for (int h = neuralNet.hiddenLayers.Length; h > -1; h--)
            {
                NeuralNetworkLayer layer;
                NeuralNetworkLayerConnection connect;
                int isize,osize;
                string ov, iv, dv;
                if (h == 0)
                {
                    iv = "is";
                    osize = KERNEL_AREA;
                }
                else
                {
                    iv = "hs" + (h - 1);
                    osize = neuralNet.hiddenLayers[h - 1].numberOfNeurons / 3;
                }
                if (h >= neuralNet.hiddenLayers.Length)
                {
                    dv = "td";
                    layer = neuralNet.outputLayer;
                    connect = neuralNet.outputConnection;
                }
                else
                {
                    dv = "ds" + (h + 1);
                    layer = neuralNet.hiddenLayers[h];
                    connect = neuralNet.hiddenConnections[h];
                }
                ov = "ds"+h;

                isize = layer.numberOfNeurons / 3;
                shaderSrc += "vec3 " + ov + "[" + osize + "];" + Environment.NewLine;

                int weightIndex = 0,
                    k = osize;
                while (k-- > 0)
                {
                    shaderSrc += ov + "[" + k + "] = (vec3(1.0,1.0,1.0)-"+iv+"["+k+"])*vec3(";

                    int c = 3;
                    while (c-- > 0)
                    {
                        int w = isize;
                        while (w-- > 0)
                        {
                            shaderSrc += "dot(" + dv + "[" + w + "],vec3(" +
                                connect.weights[weightIndex + 2].ToString(".0######") + "," +
                                connect.weights[weightIndex + 1].ToString(".0######") + "," +
                                connect.weights[weightIndex].ToString(".0######") + "))";

                            if (w != 0) shaderSrc += "+";
                            weightIndex += 3;
                        }

                        if (c != 0) shaderSrc += ",";
                        else shaderSrc += ");" + Environment.NewLine;
                    }

                    if (k == 0) shaderSrc += Environment.NewLine;
                }
            }

            return shaderSrc;
        }

        /// <summary>
        /// Compile NeuralNetwork into 1-4D shader function.
        /// </summary>
        /// <param name="nn"></param>
        /// <returns></returns>
        public static string AsShader(NeuralNetwork nn, string fname, bool glsl)
        {
            int channels = nn.outputLayer.numberOfNeurons;
            if (channels < 1 || channels > 4) return null;//invalid, max 4 output

            string chn = "";
            switch (channels) {
                case 1:
                    chn = "float";
                    break;

                case 2:
                    chn = "vec2";
                    break;

                case 3:
                    chn = "vec3";
                    break;

                case 4:
                    chn = "vec4";
                    break;
            }



            int ichannels = nn.inputLayer.numberOfNeurons;
            if (ichannels < 1 || ichannels > 4) return null;//invalid, max 4 input
            string ichn = "";
            switch (ichannels)
            {
                case 1:
                    ichn = "float";
                    break;

                case 2:
                    ichn = "vec2";
                    break;

                case 3:
                    ichn = "vec3";
                    break;

                case 4:
                    ichn = "vec4";
                    break;
            }

            string[] inNames = new string[] {"uv.x","uv.y","uv.z","uv.w"};
            string glslCode = chn+" "+fname+"("+ichn+" uv) {";


            int lastNumNeurons = ichannels,
                baseId = 0, lastId = 0, weightIndex;
            int[] stateIds = new int[nn.hiddenLayers.Length>0?nn.hiddenLayers.Length-1:0];
            for (int i = 0; i < nn.hiddenLayers.Length; i++) {
                string activeFunc = GetActivationFunctionGLSLName(nn.hiddenLayers[i].activationFunction);
                if (i != 0) {
                    if (i < nn.hiddenLayers.Length-1) stateIds[i] = baseId;
                    lastId = baseId - lastNumNeurons;
                } else {
                    if (i < stateIds.Length) stateIds[i] = 0;
                }
                int k = nn.hiddenLayers[i].numberOfNeurons;
                weightIndex = 0;
                while (k-- > 0) {
                    glslCode += "float v" + (baseId+k) + " = " + activeFunc + "(" + nn.hiddenLayers[i].biases[k].ToString(".0######")+"+";
                    if (i == 0) {
                        //input -> hidden
                        int j = lastNumNeurons;
                        while (j-- > 0)
                        {
                            glslCode += inNames[j] + "*" + nn.hiddenConnections[i].weights[weightIndex++].ToString(".0######") + (j == 0 ? "" : "+");
                        }
                        glslCode += ");";
                    } else {
                        //hidden -> hidden
                        int j = lastNumNeurons;
                        while (j-- > 0)
                        {
                            glslCode += "v" + (lastId + j) + "*" + nn.hiddenConnections[i].weights[weightIndex++].ToString(".0######") + (j == 0 ? "" : "+");
                        }
                        glslCode += ");";
                    }
                }
                lastNumNeurons = nn.hiddenLayers[i].numberOfNeurons;
                baseId += lastNumNeurons;
            }

            //hidden/input->output
            lastId = baseId - lastNumNeurons;


            string oactiveFunc = GetActivationFunctionGLSLName(nn.outputLayer.activationFunction);
            string[] ocs = new string[channels];
            int c = channels;
            weightIndex = 0;
            while (c-- > 0)
            {
                string ostr = oactiveFunc + "(" + nn.outputLayer.biases[c].ToString(".0######") + "+";

                if (nn.hiddenLayers.Length == 0)
                {
                    int j = ichannels;
                    while (j-- > 0)
                    {
                        ostr += inNames[j] + "*" + nn.outputConnection.weights[weightIndex++].ToString(".0######") + ((j == 0) ? "" : "+");
                    }
                }
                else
                {
                    int j = lastNumNeurons;
                    while (j-- > 0)
                    {
                        ostr += "v" + (lastId + j) + "*" + nn.outputConnection.weights[weightIndex++].ToString(".0######") + ((j == 0) ? "" : "+");
                    }
                }

                ocs[c] = ostr+")";
            }

            //prepare return statement and end of function
            if (channels == 1)
            {
                glslCode += "return " + ocs[0] + ";}";
            }
            else
            {
                glslCode += "return " + chn + "(";
                for (int i = 0; i < channels; i++)
                {
                    glslCode += (i==0?"":",")+ocs[i];
                }
                glslCode += ");}";
            }

            if (!glsl) return glslCode.Replace("vec", "float");
            return glslCode;
        }


        //sample 1d function points and return array of points
        /// <summary>
        /// Sample 1D function from xMin to xMax building array of 2D points.
        /// </summary>
        /// <param name="numSamples"></param>
        /// <param name="func"></param>
        /// <param name="xMin"></param>
        /// <param name="xMax"></param>
        /// <returns>Array of 2D points.</returns>
        public static float[][] SampleFunction(int numSamples, NeuralNetwork.NeuronActivationFunction func, float xMin, float xMax)
        {
            float[][] res = new float[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                float sx = (i / (float)(numSamples - 1)) * (xMax - xMin) + xMin;

                res[i] = new float[] {
                    sx,
                    func(sx)
                };
            }
            return res;
        }


        /// <summary>
        /// Sample 1D neural network from xMin to xMax building array of 2D points.
        /// </summary>
        /// <param name="numSamples"></param>
        /// <param name="nn"></param>
        /// <param name="xMin"></param>
        /// <param name="xMax"></param>
        /// <returns></returns>
        public static float[][] SampleNeuralNetwork(int numSamples, NeuralNetwork nn, float xMin, float xMax)
        {
            NeuralNetworkProgram nnp = new NeuralNetworkProgram(nn);
            nnp.context.Reset(true);

            float[][] res = new float[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                float sx = (i / (float)(numSamples - 1)) * (xMax - xMin) + xMin;

                nnp.context.inputData[0] = sx;
                nnp.Execute();

                res[i] = new float[] {
                    sx,
                    nnp.context.outputData[0]
                };
            }
            return res;
        }


        /// <summary>
        /// Get default activation function name from function.
        /// </summary>
        /// <param name="activeFunc"></param>
        /// <returns>Name of activation function.</returns>
        public static string GetActivationFunctionName(NeuralNetwork.NeuronActivationFunction activeFunc)
        {
            if (activeFunc == Identity_ActivationFunction) return "identity";
            if (activeFunc == Rectifier_ActivationFunction) return "rectifier";
            if (activeFunc == Sin_ActivationFunction) return "sin";
            if (activeFunc == Cos_ActivationFunction) return "cos";
            if (activeFunc == Tan_ActivationFunction) return "tan";
            if (activeFunc == Tanh_ActivationFunction) return "tanh";
            if (activeFunc == Sinh_ActivationFunction) return "sinh";
            if (activeFunc == Exp_ActivationFunction) return "exp";
            if (activeFunc == Sigmoid_ActivationFunction) return "sigmoid";
            if (activeFunc == Sqrt_ActivationFunction) return "sqrt";
            if (activeFunc == Pow2_ActivationFunction) return "pow2";
            return "unknown";
        }

        public static string GetActivationFunctionGLSLName(NeuralNetwork.NeuronActivationFunction activeFunc)
        {
            if (activeFunc == Identity_ActivationFunction) return "";
            if (activeFunc == Rectifier_ActivationFunction) return "rectifier";
            if (activeFunc == Sin_ActivationFunction) return "sin";
            if (activeFunc == Cos_ActivationFunction) return "cos";
            if (activeFunc == Tan_ActivationFunction) return "tan";
            if (activeFunc == Tanh_ActivationFunction) return "tanh";
            if (activeFunc == Sinh_ActivationFunction) return "sinh";
            if (activeFunc == Exp_ActivationFunction) return "exp";
            if (activeFunc == Sigmoid_ActivationFunction) return "sigmoid";
            if (activeFunc == Sqrt_ActivationFunction) return "sqrt";
            if (activeFunc == Pow2_ActivationFunction) return "pow2";
            return "";
        }

        public static int GetActivationFunctionID(NeuralNetwork.NeuronActivationFunction activeFunc)
        {
            if (activeFunc == Identity_ActivationFunction) return 0;
            if (activeFunc == Rectifier_ActivationFunction) return 1;
            if (activeFunc == Sin_ActivationFunction) return 2;
            if (activeFunc == Cos_ActivationFunction) return 3;
            if (activeFunc == Tan_ActivationFunction) return 4;
            if (activeFunc == Tanh_ActivationFunction) return 5;
            if (activeFunc == Sinh_ActivationFunction) return 6;
            if (activeFunc == Exp_ActivationFunction) return 7;
            if (activeFunc == Sigmoid_ActivationFunction) return 8;
            if (activeFunc == Sqrt_ActivationFunction) return 9;
            if (activeFunc == Pow2_ActivationFunction) return 10;
            return -1;
        }

        public static NeuralNetwork.NeuronActivationFunction GetActivationFunctionFromID(int i)
        {
            switch (i)
            {
                case 0: return Identity_ActivationFunction;

                case 1: return Rectifier_ActivationFunction;

                case 2: return Sin_ActivationFunction;

                case 3: return Cos_ActivationFunction;

                case 4: return Tan_ActivationFunction;

                case 5: return Tanh_ActivationFunction;

                case 6: return Sinh_ActivationFunction;

                case 7: return Exp_ActivationFunction;

                case 8: return Sigmoid_ActivationFunction;

                case 9: return Sqrt_ActivationFunction;

                case 10: return Pow2_ActivationFunction;

            }

            return Identity_ActivationFunction;
        }

        //premade activation functions
        /// <summary>
        /// Identity(no activation function), linear.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Identity_ActivationFunction(float v)
        {
            return v;
        }

        /// <summary>
        /// Rectifier activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Rectifier_ActivationFunction(float v)
        {
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }

        /// <summary>
        /// Unclamped exp activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Exp_ActivationFunction(float v)
        {
            v = (float)Math.Exp(v);
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            return v;
        }

        /// <summary>
        /// Sigmoid activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Sigmoid_ActivationFunction(float v)
        {
            v = 1.0f / (1.0f + (float)Math.Exp(-v));
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }
        /// <summary>
        /// Sin activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Sin_ActivationFunction(float v)
        {
            v = (float)Math.Sin(v);
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }
        /// <summary>
        /// Cos activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Cos_ActivationFunction(float v)
        {
            v = (float)Math.Cos(v);
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }
        /// <summary>
        /// Tan activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Tan_ActivationFunction(float v)
        {
            v = (float)Math.Tan(v);
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }
        /// <summary>
        /// Tanh activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Tanh_ActivationFunction(float v)
        {
            v = (float)Math.Tanh(v);
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }
        /// <summary>
        /// Sinh activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Sinh_ActivationFunction(float v)
        {
            v = (float)Math.Sinh(v);
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }

        /// <summary>
        /// Sqrt activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Sqrt_ActivationFunction(float v)
        {
            v = (float)Math.Sqrt(v);
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }

        /// <summary>
        /// Pow2 activation function.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static float Pow2_ActivationFunction(float v)
        {
            v = v * v;
#pragma warning disable 414,1718
            if (v != v) return 0.0f;
#pragma warning restore 1718
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }


        private static Random random = new Random();
        /// <summary>
        /// Random float(0-1), not thread safe.
        /// </summary>
        /// <returns>Random float(0-1).</returns>
        public static float NextFloat01()
        {
            return (float)random.NextDouble();
        }

        /// <summary>
        /// Random int between min and max, not thread safe.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static int NextInt(int min, int max)
        {
            return random.Next(max - min) + min;
        }

        public static void IntToStream(int i, Stream s)
        {
            s.Write(BitConverter.GetBytes(IPAddress.HostToNetworkOrder(i)), 0, 4);
        }
        public static int IntFromStream(Stream s)
        {
            byte[] b = new byte[4];
            s.Read(b, 0, 4);
            return IPAddress.NetworkToHostOrder(BitConverter.ToInt32(b, 0));
        }
        //convert int and float to bytes and big endian
        /// <summary>
        /// Convert int to bytes.
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public static byte[] IntToBytes(int i)
        {
            return BitConverter.GetBytes(IPAddress.HostToNetworkOrder(i));
        }
        /// <summary>
        /// Convert array of bytes to int.
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        public static int IntFromBytes(byte[] b)
        {
            return IPAddress.NetworkToHostOrder(BitConverter.ToInt32(b, 0));
        }
        /// <summary>
        /// Convert array of bytes to int.
        /// </summary>
        /// <param name="b"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public static int IntFromBytes(byte[] b, int index)
        {
            return IPAddress.NetworkToHostOrder(BitConverter.ToInt32(b, index));
        }
        /// <summary>
        /// Convert float to bytes.
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        public static byte[] FloatToBytes(float f)
        {
            byte[] b = BitConverter.GetBytes(f);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(b);
            }
            return b;
        }
        public static void FloatToStream(float f, Stream s)
        {
            byte[] b = BitConverter.GetBytes(f);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(b);
            }
            s.Write(b, 0, 4);
        }
        /// <summary>
        /// Convert array of bytes to float.
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        public static float FloatFromBytes(byte[] b)
        {
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(b, 0, 4);
            }
            return BitConverter.ToSingle(b, 0);
        }
        /// <summary>
        /// Convert array of bytes to float.
        /// </summary>
        /// <param name="b"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public static float FloatFromBytes(byte[] b, int index)
        {
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(b, index, 4);
            }
            return BitConverter.ToSingle(b, index);
        }

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		/// <param name="s">S.</param>
        public static float FloatFromStream(Stream s)
        {
            byte[] b = new byte[4];
            s.Read(b, 0, 4);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(b, 0, 4);
            }
            return BitConverter.ToSingle(b, 0);
        }

		/// <summary>
		/// 
		/// </summary>
		/// <param name="f"></param>
		/// <param name="s"></param>
        public static void FloatArrayToStream(float[] f, Stream s)
        {
            IntToStream(f.Length, s);
            int k = f.Length;
            while (k-- > 0)
            {
                FloatToStream(f[k], s);
            }
        }

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		/// <param name="s"></param>
        public static float[] FloatArrayFromStream(Stream s)
        {
            float[] f = new float[IntFromStream(s)];
            int k = f.Length;
            while (k-- > 0)
            {
                f[k] = FloatFromStream(s);
            }
            return f;
        }

		/// <summary>
		/// 
		/// </summary>
		/// <param name="f"></param>
		/// <param name="s"></param>
		public static void FloatArrayFromStream(float[] f, Stream s)
		{
			int k = f.Length;
			while (k-- > 0)
			{
				f[k] = FloatFromStream(s);
			}
		}

        /// <summary>
        /// Lerp from a to b over t.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public static float Lerp(float a, float b, float t)
        {
            return a + (b - a) * t;
        }

        //fill float array with single value
        /// <summary>
        /// Fill float array with single value.
        /// </summary>
        /// <param name="f"></param>
        /// <param name="v"></param>
        public static void Fill(float[] f, float v)
        {
            int i = f.Length;
            while (i-- > 0)
            {
                f[i] = v;
            }
        }

        public static void Multiply(float[] f, float v)
        {
            int i = f.Length;
            while (i-- > 0)
            {
                f[i] *= v;
            }
        }

        //find largest value from array of values and return index
        /// <summary>
        /// Returns index of largest value in array.
        /// </summary>
        /// <param name="f"></param>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns>Index of largest value.</returns>
        public static int Largest(float[] f, int start, int end)
        {
            int k = end,
                i = end - 1;
            float l = float.MinValue;
            while (k-- > start)
            {
                if (f[k] > l)
                {
                    l = f[k];
                    i = k;
                }
            }
            return i;
        }
        /// <summary>
        /// Random int from array of probabilities.
        /// </summary>
        /// <param name="p"></param>
        /// <returns>Index of selected probability.</returns>
        public static int RandomChoice(float[] p)
        {
            float rand = NextFloat01(),
                  sum = 0.0f;
            for (int i = 0; i < p.Length; i++)
            {
                float prob = p[i],
                      nsum = sum + prob;
                if (rand >= sum && rand <= nsum)
                {
                    return i;
                }
                sum = nsum;
            }
            return p.Length - 1;
        }

        /// <summary>
        /// Normalize vector.
        /// </summary>
        /// <param name="f"></param>
        public static void Normalize(float[] f)
        {
            float sum = 0.0f;
            int i = f.Length;
            while (i-- > 0)
            {
                sum += f[i];
            }
            if (sum > 0.0f)
            {
                i = f.Length;
                while (i-- > 0)
                {
                    f[i] /= sum;
                }
            }
        }

        /// <summary>
        /// Puts probabilities to power of pow and re-normalizes.
        /// </summary>
        /// <param name="p"></param>
        /// <param name="pow"></param>
        public static void ProbabilityPower(float[] p, float pow)
        {
            float sum = 0.0f;
            int i = p.Length;
            while (i-- > 0)
            {
                p[i] = (float)Math.Pow(p[i], pow);
                sum += p[i];
            }
            i = p.Length;
            while (i-- > 0)
            {
                p[i] /= sum;
            }
        }



        /// <summary>
        /// Encode multiple strings into one-hot encoding and outputs dictionary of used characters.
        /// </summary>
        /// <param name="txt"></param>
        /// <param name="dict"></param>
        /// <returns></returns>
        public static float[][][] EncodeStringOneHot(string[] atxt, out List<char> dict)
        {
            dict = new List<char>();
            for (int a = 0; a < atxt.Length; a++)
            {
                string txt = atxt[a];
                for (int i = 0; i < txt.Length; i++)
                {
                    if (!dict.Contains(txt[i]))
                    {
                        dict.Add(txt[i]);
                    }
                }
            }

            float[][][] output = new float[atxt.Length][][];
            for (int a = 0; a < atxt.Length; a++)
            {
                string txt = atxt[a];
                output[a] = new float[txt.Length][];
                for (int i = 0; i < txt.Length; i++)
                {
                    output[a][i] = new float[dict.Count];
                    Fill(output[a][i], 0.0f);
                    output[a][i][dict.IndexOf(txt[i])] = 1.0f;
                }
            }

            return output;
        }

        /// <summary>
        /// Encode string into one-hot encoding and outputs dictionary of used characters.
        /// </summary>
        /// <param name="txt"></param>
        /// <param name="dict"></param>
        /// <returns></returns>
        public static float[][] EncodeStringOneHot(string txt, out List<char> dict)
        {
            dict = new List<char>();
            for (int i = 0; i < txt.Length; i++)
            {
                if (!dict.Contains(txt[i]))
                {
                    dict.Add(txt[i]);
                }
            }

            float[][] output = new float[txt.Length][];
            for (int i = 0; i < txt.Length; i++)
            {
                output[i] = new float[dict.Count];
                Fill(output[i], 0.0f);
                output[i][dict.IndexOf(txt[i])] = 1.0f;
            }
            return output;
        }
        /// <summary>
        /// Decode one-hot encoded string using dictionary.
        /// </summary>
        /// <param name="txt"></param>
        /// <param name="dict"></param>
        /// <returns></returns>
        public static string DecodeStringOneHot(float[][] txt, List<char> dict)
        {
            string decoded = "";
            for (int i = 0; i < txt.Length; i++)
            {
                decoded += dict[Largest(txt[i], 0, txt[i].Length)];
            }
            return decoded;
        }


        /// <summary>
        /// Shuffle array.
        /// </summary>
        /// <param name="a"></param>
        public static void Shuffle(Array a)
        {
            int l = a.Length,
                i = l;
            while (i-- > 0)
            {
                int r = NextInt(0,l),
                    w = NextInt(0,l);
                object temp = a.GetValue(w);
                a.SetValue(a.GetValue(r), w);
                a.SetValue(temp, r);
            }
        }
        public static void Shuffle(Array a, Array b)
        {
            int l = a.Length,
                i = l;
            while (i-- > 0)
            {
                int r = NextInt(0, l),
                    w = NextInt(0, l);
                object temp = a.GetValue(w);
                a.SetValue(a.GetValue(r), w);
                a.SetValue(temp, r);

                temp = b.GetValue(w);
                b.SetValue(b.GetValue(r), w);
                b.SetValue(temp, r);
            }
        }
    }
}
