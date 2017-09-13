﻿using System;
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
                "1 8 2018211316 5 good",
                "1 4 89121211825 7 good",
                "0 6 112524 4 bad",
                "0 7 41821475 6 bad",
                "1 2 20152351819 1 good",
                "1 1 1915181519 5 good",
                "0 18 18544920 6 bad",
                "0 10 16151920 4 bad",
                "1 3 918131 4 good",
                "1 8 8211818931145 9 good",
                "1 3 195318520 6 other",
                "0 5 7154 3 other"
            };

            Algotelli.Run(rawData, false);
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
