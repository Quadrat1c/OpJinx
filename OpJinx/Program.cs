using System;
using System.Collections.Generic;
using System.Linq;

namespace OpJinx
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Operation Jinx\r\n");

            // Just test data for Algotelli.
            string[] rawData = new string[]
            {
                "1 8 485 5 good",
                "1 4 24233 7 good",
                "0 6 7392 4 bad",
                "0 7 543 6 bad",
                "1 2 245 1 good",
                "1 1 2468 5 good",
                "0 18 782 6 bad",
                "0 10 425 4 bad",
                "1 3 4425 4 good",
                "1 8 456681 9 good",
                "1 3 44567 6 other",
                "0 5 4204 3 other"
                /*
                "1 8 trump 5 good",
                "1 4 hillary 7 good",
                "0 6 alex 4 bad",
                "0 7 drudge 6 bad",
                "1 2 towers 1 good",
                "1 1 soros 5 good",
                "0 18 reddit 6 bad",
                "0 10 post 4 bad",
                "1 3 irma 4 good",
                "1 8 hurricane 9 good",
                "1 3 secret 6 other",
                "0 5 god 3 other"
                */
                // TODO: Convert all text to uniform numbers that will best represent words
                // and keep the integrity of the neural networks accuracy.
            };

            Algotelli.Run(rawData, true);
            Console.ReadLine();

            List<string> keywords = new List<string>();

            Console.WriteLine("abovetopsecret.com:");
            keywords = Words.DensityCheck("http://abovetopsecret.com", false);
            DisplayKeyWords(keywords, 2);
            Console.ReadLine();

            Console.WriteLine("godlikeproductions.com:");
            keywords = Words.DensityCheck("http://godlikeproductions.com", false);
            DisplayKeyWords(keywords, 2);
            Console.ReadLine();

            Console.WriteLine("infowars.com:");
            keywords = Words.DensityCheck("http://infowars.com", false);
            DisplayKeyWords(keywords, 2);
            Console.ReadLine();

            Console.WriteLine("drudgereport.com:");
            keywords = Words.DensityCheck("http://drudgereport.com/", false);
            DisplayKeyWords(keywords, 2);
            Console.ReadLine();

            Console.WriteLine("reddit.com:");
            keywords = Words.DensityCheck("http://reddit.com/", false);
            DisplayKeyWords(keywords, 2);
            Console.ReadLine();

            Console.WriteLine("cnn.com:");
            keywords = Words.DensityCheck("http://cnn.com", false);
            DisplayKeyWords(keywords, 2);

            // TODO: Place the key words and density into a string[] to send to algotelli.

            Console.Write("Press any key to exit.");
            Console.ReadLine();
        }

        /// <summary>
        /// input a list and a minimum count for keyword density
        /// </summary>
        /// <param name="list">The list you wanto display</param>
        /// <param name="minCount">Minimum count for keyword density</param>
        static void DisplayKeyWords(List<string> list, int minCount)
        {
            var keywords = list.GroupBy(x => x).OrderByDescending(x => x.Count());

            foreach (var word in keywords)
            {
                if (word.Count() > minCount)
                {
                    Console.WriteLine($"{word.Key} {word.Count()}");
                }
            }
        }
    }
}
