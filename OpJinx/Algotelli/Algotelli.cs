using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/*
 * Algotelli is a algorithm that encodes
 * Normalized Data to Numerical Values
 * to use as training data for our neural network.
 */

namespace OpJinx
{
    public class Algotelli
    {
        // TODO: Tidy up console outputs.
        public static void Run(string[] rawData, bool debug)
        {
            Console.WriteLine("Algotelli - Encoding normalized data to numerical values");
            #region Debug
            if (debug == true)
            {
                Console.WriteLine("\r\nRaw Data:");
                ShowRaw(rawData);
            }
            #endregion

            string[] colTypes = new string[5] {
                "binary",           // Vector (Seen on different sites)
                "numeric",          // Density
                "numeric",          // Word Encoded
                "numeric",          // Word Length
                "categorical"       // Result
            };
            #region Debug
            if (debug == true)
            {
                Console.WriteLine("\r\nColumn Types:");
                ShowCol(colTypes);
            }
            #endregion

            Console.WriteLine("Begin Transform");
            double[][] nnData = Transform(rawData, colTypes, debug);   // neural network data
            Console.WriteLine("Transform Complete");

            Console.WriteLine("\r\nResult Data:");
            ShowTransformed(nnData);
        }

        static double[][] Transform(string[] rawData, string[] colTypes, bool debug)
        {
            // tokenize raw data into a string matrix
            Console.WriteLine("[!] Loading Data.");
            string[][] data = LoadData(rawData);
            #region Debug
            if (debug == true)
            {
                Console.WriteLine("\r\nTokenized Data:");
                ShowToken(data);
            }
            #endregion
            Console.WriteLine("[+] Data Load Complete.");

            // extract distinct values
            Console.WriteLine("[!] Scanning Data for Distinct Values.");
            string[][] distinctValues = GetValues(data, colTypes);  // includes binary
            #region Debug
            if (debug == true)
            {
                Console.WriteLine("\r\nScanning tokenized data to extract distinct values:");
                ShowDistinctValues(distinctValues);
            }
            #endregion
            Console.WriteLine("[+] Extracted Distinct Values.");

            // compute number of columns for result matrix
            Console.WriteLine("[!] Computing number of columns for matrix.");
            int extraCols = NumNewCols(distinctValues, colTypes);   // binary does not add any
            #region Debug
            if (debug == true)
            {
                Console.WriteLine("\r\nComputing number of columns for result matrix.");
                Console.WriteLine($"Adding {extraCols} columns for categorical data encoding.");
            }
            #endregion
            Console.WriteLine("[+] Columns added to matrix.");

            double[][] result = new double[data.Length][];
            for (int i = 0; i < result.Length; ++i)
            {
                result[i] = new double[data[0].Length + extraCols];
            }

            Console.WriteLine("\nComputing means and standard deviations of numeric data");
            double[] means = GetMeans(data, colTypes);
            double[] stdDevs = GetStdDevs(data, colTypes, means);
            Console.WriteLine("\nMeans:");
            ShowVector(means, 2);
            Console.WriteLine("\nStandard deviations:");
            ShowVector(stdDevs, 2);

            Console.WriteLine("\nEntering main transform loop");
            for (int row = 0; row < data.Length; ++row)
            {
                int k = 0;  // walk across result cols
                for (int col = 0; col < data[row].Length; ++col)
                {
                    string val = data[row][col];
                    bool isBinary = (colTypes[col] == "binary");
                    bool isCategorical = (colTypes[col] == "categorical");
                    bool isNumeric = (colTypes[col] == "numeric");
                    bool isIndependent = (col < data[0].Length - 1);
                    bool isDependent = (col == data[0].Length - 1);

                    // binary x value -> -1.0 or +1.0
                    if (isBinary && isIndependent)
                    {
                        result[row][k++] = BinaryIndepenToValue(val, col, distinctValues);
                    }
                    // binary y value -> 0.0 or 1.0
                    else if (isBinary && isDependent)
                    {
                        result[row][k] = BinaryDepenToValue(val, col, distinctValues);  // no k++
                    }
                    // cat x value -> [0.0, 1.0, 1.0] or [-1.0, -1.0, -1.0]
                    else if (isCategorical && isIndependent)
                    {
                        double[] vals = CatIndepenToValues(val, col, distinctValues);
                        for (int j = 0; j < vals.Length; ++j)
                            result[row][k++] = vals[j];
                    }
                    // cat y value -> [1.0, 0.0, 0.0]
                    else if (isCategorical && isDependent)
                    {
                        double[] vals = CatDepenToValues(val, col, distinctValues);
                        for (int j = 0; j < vals.Length; ++j)
                            result[row][k++] = vals[j];
                    }
                    else if (isNumeric && isIndependent)
                    {
                        result[row][k++] = NumIndepenToValue(val, col, means, stdDevs);
                    }
                    else if (isNumeric && isDependent)
                    {
                        result[row][k] = double.Parse(val); // no k++
                    }
                }
            }
            return result;
        }

        // Load Raw Data
        static string[][] LoadData(string[] rawData)
        {
            int numRows = rawData.Length;
            int numCols = rawData[0].Split(' ').Length;
            string[][] result = new string[numRows][];

            for (int i = 0; i < numRows; ++i)
            {
                result[i] = new string[numCols];
                string[] tokens = rawData[i].Split(' ');
                Array.Copy(tokens, result[i], numCols);
            }

            return result;
        }

        #region Transform Data
        // binary x value -> -1 or +1
        static double BinaryIndepenToValue(string val, int col, string[][] distinctValues)
        {
            if (distinctValues[col].Length != 2)
                throw new Exception("Binary x data only 2 values allowed");
            if (distinctValues[col][0] == val)
                return -1.0;
            else
                return +1.0;
        }

        // binary y value -> 0 or 1
        static double BinaryDepenToValue(string val, int col, string[][] distinctValues)
        {
            if (distinctValues[col].Length != 2)
                throw new Exception("Binary y data only 2 values allowed");
            if (distinctValues[col][0] == val)
                return 0.0;
            else
                return 1.0;
        }

        // categorical x value -> 1-of-(C-1) effects encoding
        static double[] CatIndepenToValues(string val, int col, string[][] distinctValues)
        {
            if (distinctValues[col].Length == 2)
                throw new Exception("Categorical x data only 1, 3+ values allowed");
            int size = distinctValues[col].Length;
            double[] result = new double[size];

            int idx = 0;
            for (int i = 0; i < size; ++i)
            {
                if (distinctValues[col][i] == val)
                {
                    idx = i; break;
                }
            }

            if (idx == size - 1) // the value is the last one so use effects encoding
            {
                for (int i = 0; i < size; ++i) // ex: [-1.0, -1.0, -1.0]
                {
                    result[i] = -1.0;
                }
            }
            else // value is not last, use dummy
            {
                result[result.Length - 1 - idx] = +1.0; // ex: [0.0, 1.0, 0.0]
            }
            return result;
        }

        // categorical y value -> 1-of-C dummy encoding
        static double[] CatDepenToValues(string val, int col, string[][] distinctValues)
        {
            if (distinctValues[col].Length == 2)
                throw new Exception("Categorical x data only 1, 3+ values allowed");
            int size = distinctValues[col].Length;
            double[] result = new double[size];

            int idx = 0;
            for (int i = 0; i < size; ++i)
            {
                if (distinctValues[col][i] == val)
                {
                    idx = i; break;
                }
            }
            result[result.Length - 1 - idx] = 1.0; // ex: [0.0, 1.0, 0.0]
            return result;
        }

        // numeric x value -> (x - m) / s
        static double NumIndepenToValue(string val, int col, double[] means, double[] stdDevs)
        {
            double x = double.Parse(val);
            double m = means[col];
            double sd = stdDevs[col];
            return (x - m) / sd;
        }

        static int NumNewCols(string[][] distinctValues, string[] colTypes)
        {
            // number of additional columns needed due to categorical encoding
            int result = 0;
            for (int i = 0; i < colTypes.Length; ++i)
            {
                if (colTypes[i] == "categorical")
                {
                    int numCatValues = distinctValues[i].Length;
                    result += (numCatValues - 1);
                }
            }
            return result;
        }
        #endregion

        #region Get Data
        // Examine tokenized data to get distinct values for cat and binary columns
        static string[][] GetValues(string[][] data, string[] colTypes)
        {
            int numCols = data[0].Length;
            string[][] result = new string[numCols][];

            for (int col = 0; col < numCols; ++col)
            {
                if (colTypes[col] == "numeric")
                {
                    result[col] = new string[] { "numeric" };
                }
                else
                {
                    Dictionary<string, bool> d = new Dictionary<string, bool>();    // bool is a dummy

                    for (int row = 0; row < data.Length; ++row)
                    {
                        string currVal = data[row][col];
                        if (d.ContainsKey(currVal) == false)
                            d.Add(currVal, true);
                    }

                    result[col] = new string[d.Count];
                    int k = 0;

                    foreach (string val in d.Keys)
                        result[col][k++] = val;
                }
            }

            return result;
        }

        static double[] GetMeans(string[][] data, string[] colTypes)
        {
            double[] result = new double[data.Length];
            for (int col = 0; col < data[0].Length; ++col)  // each column
            {
                if (colTypes[col] != "numeric") continue;   // curr col is not numeric

                double sum = 0.0;
                for (int row = 0; row < data.Length; ++row)
                {
                    double val = double.Parse(data[row][col]);
                    sum += val;
                }
                result[col] = sum / data.Length;
            }
            return result;
        }

        static double[] GetStdDevs(string[][] data, string[] colTypes, double[] means)
        {
            double[] result = new double[data.Length];
            for (int col = 0; col < data[0].Length; ++col) // each column
            {
                if (colTypes[col] != "numeric") continue; // curr col is not numeric

                double sum = 0.0;
                for (int row = 0; row < data.Length; ++row)
                {
                    double val = double.Parse(data[row][col]);
                    sum += (val - means[col]) * (val - means[col]);
                }
                result[col] = Math.Sqrt(sum / data.Length);
            }
            return result;
        }
        #endregion

        #region Show Data
        // Show Raw Data
        static void ShowRaw(string[] rawData)
        {
            for (int i = 0; i < rawData.Length; ++i)
            {
                Console.WriteLine(rawData[i]);
            }
        }

        // Show Column Types
        static void ShowCol(string[] colTypes)
        {
            for (int i = 0; i < colTypes.Length; ++i)
            {
                Console.WriteLine(colTypes[i] + " ");
            }
            Console.WriteLine("");
        }

        // Show Tokenized Data
        static void ShowToken(string[][] data)
        {
            Console.WriteLine("Multi  Density   Word            Length         -> Result");
            Console.WriteLine("-------.---------.---------------.--------------.-----------------");

            for (int i = 0; i < data.Length; ++i)
            {
                Console.WriteLine($"Data Length: {data[i].Length}");
                for (int j = 0; j < data[i].Length; ++j)
                {
                    if (j == 4) Console.Write(" -> ");
                    Console.Write(data[i][j].PadRight(10) + " ");
                }
                Console.WriteLine("");
            }
        }

        // Show Distinct Values
        static void ShowDistinctValues(string[][] distinctvalues)
        {
            for (int i = 0; i < distinctvalues.Length; ++i)
            {
                Console.Write($"[{i}]");

                for (int j = 0; j < distinctvalues[i].Length; ++j)
                {
                    Console.Write(distinctvalues[i][j] + " ");
                }
                Console.WriteLine("");
            }
        }

        // Show Vectors
        static void ShowVector(double[] vector, int decimals)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % 12 == 0) // max of 12 values per row
                    Console.WriteLine("");
                if (vector[i] >= 0.0) Console.Write(" ");
                Console.Write(vector[i].ToString($"F {decimals} ")); // 2 decimals
            }

            Console.WriteLine("\n");
        }

        // Show Transformed Data
        static void ShowTransformed(double[][] nnData)
        {
            Console.WriteLine("Multi  Density   Word            Length         -> Result");
            Console.WriteLine("--------------------------------------------------------------");
            for (int i = 0; i < nnData.Length; ++i)
            {
                for (int j = 0; j < nnData[i].Length; ++j)
                {
                    if (j == 6) Console.Write("-> ");
                    if (nnData[i][j] >= 0.0) Console.Write(" ");
                    Console.Write(nnData[i][j].ToString("F2") + "   ");
                }
                Console.WriteLine("");
            }
        }
        #endregion
    }
}
