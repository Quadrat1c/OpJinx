using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpJinx
{
    public class Commands
    {
        public static List<string> encodedKeywords = new List<string>();

        public static void ListCommands()
        {
            Console.WriteLine("[Main]");
            CmdColor("help", "displays a list of commands.");
            CmdColor("version", "displays current version.");
            CmdColor("clear", "clears console.");
            CmdColor("exit", "exits the program.");
            Console.WriteLine("");
            Console.WriteLine("[Jinx]");
            CmdColor("keywords", "get keywords from sites.");
            CmdColor("nnet", "neural net test.");
            CmdColor("aglotelli", "runs algotelli test.");
            CmdColor("web", "runs web scraper test.");
            Console.WriteLine("");
            Console.WriteLine("[Utility]");
            CmdColor("dec", "convert a word to decimal format for nnet.");
        }

        private static void CmdColor(string cmd, string desc)
        {
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.Write($"    {cmd}");
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.Write(" - {0}\r\n", desc);
            Console.ForegroundColor = ConsoleColor.Red;
        }

        #region Keywords
        public static void GetKeywords()
        {
            // TODO: Create a list of urls to scan instead of the repetitive stuff below.
            List<string> keywords = new List<string>();

            Console.WriteLine("abovetopsecret.com:");
            keywords = Words.DensityCheck("http://abovetopsecret.com", false);
            DisplayKeyWords(keywords, 2);
            Continue();
            Console.ReadLine();

            Console.WriteLine("godlikeproductions.com:");
            keywords = Words.DensityCheck("http://godlikeproductions.com", false);
            DisplayKeyWords(keywords, 2);
            Console.ReadLine();

            Console.WriteLine("infowars.com:");
            keywords = Words.DensityCheck("http://infowars.com", false);
            DisplayKeyWords(keywords, 2);
            Continue();
            Console.ReadLine();

            Console.WriteLine("drudgereport.com:");
            keywords = Words.DensityCheck("http://drudgereport.com/", false);
            DisplayKeyWords(keywords, 2);
            Continue();
            Console.ReadLine();

            Console.WriteLine("reddit.com:");
            keywords = Words.DensityCheck("http://reddit.com/", false);
            DisplayKeyWords(keywords, 2);
            Continue();
            Console.ReadLine();

            Console.WriteLine("cnn.com:");
            keywords = Words.DensityCheck("http://cnn.com", false);
            DisplayKeyWords(keywords, 2);

            var encodedKeys = encodedKeywords.GroupBy(x => x).OrderByDescending(x => x.Count());

            foreach (var word in encodedKeys)
            {
                Console.WriteLine($"{word.Key}");
            }

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("End of keyword collection.");
        }

        private static void Continue()
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Press any key to continue.");
            Console.ForegroundColor = ConsoleColor.Red;
        }

        /// <summary>
        /// input a list and a minimum count for keyword density
        /// </summary>
        /// <param name="list">The list you wanto display</param>
        /// <param name="minCount">Minimum count for keyword density</param>
        private static void DisplayKeyWords(List<string> list, int minCount)
        {
            var keywords = list.GroupBy(x => x).OrderByDescending(x => x.Count());

            foreach (var word in keywords)
            {
                if (word.Count() > minCount)
                {
                    string[] input = new string[] { word.Key, word.Count().ToString() };
                    Console.WriteLine($"{word.Key} {word.Count()}");
                    encodedKeywords.InsertRange(2, input);
                    //encodedKeywords.Add("0." + Ciphers.Text2Num(word.Key) + " " + word.Count());
                }
            }
        }
        #endregion

        #region NeuralNet
        public static void NeuralNetTest()
        {
            new NeuralNet();
        }
        #endregion

        #region Algotelli
        public static void AlgotelliTest()
        {
            string[] rawData = new string[]
           {
                "1 8 0.2018211316 5 good",
                "1 4 0.89121211825 7 good",
                "0 6 0.112524 4 bad",
                "0 7 0.41821475 6 bad",
                "1 2 0.20152351819 1 good",
                "1 1 0.1915181519 5 good",
                "0 18 0.18544920 6 bad",
                "0 10 0.16151920 4 bad",
                "1 3 0.918131 4 good",
                "1 8 0.8211818931145 9 good",
                "1 3 0.195318520 6 other",
                "0 5 0.7154 3 other"
           };

            Console.WriteLine("Debug type ON or OFF");
            Console.ForegroundColor = ConsoleColor.Green;

            if (Console.ReadLine().ToLower().Contains("on"))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Algotelli.Run(rawData, true);
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Algotelli.Run(rawData, false);
            }
        }
        #endregion
    }
}
