using System;
using System.Collections.Generic;

namespace OpJinx
{
    class Program
    {
        static void Main(string[] args)
        {
            // Testing github commit
            string ver = "0.2.2";
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"  _,-._");
            Console.WriteLine(@" / \_/ \");
            Console.WriteLine($" >-(_)-< ");
            Console.WriteLine(@" \_/ \_/");
            Console.WriteLine($"   `-'   Operation Jinx (v{ver})");
            Console.WriteLine("__________________________________\r\n");
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
                    case "web":
                        Console.ForegroundColor = ConsoleColor.Red;
                        Web.Crawler.Start();
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
