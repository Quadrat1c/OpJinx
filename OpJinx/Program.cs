using System;
using System.Collections.Generic;
using System.Linq;

namespace OpJinx
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Operation Jinx");

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
