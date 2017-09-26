using System;
using System.Collections.Generic;
using System.Linq;

namespace OpJinx
{
    class Program
    {
        static void Main(string[] args)
        {
            string ver = "0.2.1";
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Operation Jinx (v{ver})");
            Console.WriteLine("_______________________\r\n");
            Console.ForegroundColor = ConsoleColor.DarkRed;
            Console.WriteLine($"Type 'help' for a list of commands.");

            bool quitNow = false;
            string cmd;

            while (!quitNow)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                cmd = Console.ReadLine();

                switch (cmd)
                {
                    // Main ********************************************
                    case "help":
                    case "'help'":
                    case "h":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Commands.ListCommands();
                        Console.WriteLine("");
                        break;

                    case "ver":
                    case "version":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine($"Version: {ver}");
                        Console.WriteLine("");
                        break;

                    case "clear":
                        Console.Clear();
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine($"Operation Jinx (v{ver})");
                        Console.WriteLine("_______________________\r\n");
                        Console.WriteLine("");
                        break;

                    case "exit":
                    case "quit":
                    case "q":
                        Environment.Exit(0);
                        break;

                    // Jinx ********************************************
                    case "keywords":
                    case "kw":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Commands.GetKeywords();
                        Console.WriteLine("");
                        break;

                    case "nnet":
                    case "neuralnet":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Commands.NeuralNetTest();
                        Console.WriteLine("");
                        break;

                    case "algotelli":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Commands.AlgotelliTest();
                        Console.WriteLine("");
                        break;

                    case "web":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Web.Crawler.Start();
                        Console.WriteLine("");
                        break;

                    // Utility ********************************************
                    case "dec":
                        // TODO: Convert words to decimal using this concept.
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.Write("Enter word to convert: ");
                        string word = Ciphers.Text2Num(Console.ReadLine());
                        word = "0." + word;
                        double test = Convert.ToDouble(word);
                        Console.WriteLine(test);
                        Console.WriteLine("");
                        break;

                    // Default ********************************************
                    default:
                        Console.ForegroundColor = ConsoleColor.DarkRed;
                        Console.WriteLine($"Unknown command: {cmd}");
                        Console.WriteLine("");
                        break;
                }
            }
        }  
    }
}