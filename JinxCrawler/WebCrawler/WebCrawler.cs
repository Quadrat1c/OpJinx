using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using HtmlAgilityPack;
using System.Text.RegularExpressions;

namespace OpJinx.Web
{
    public class Crawler
    {
        public static void Start()
        {
            CrawlSite("http://abovetopsecret.com", "//div[@class='headline']");
            CrawlSite("http://infowars.com", "//div[@class='article-content']");
            CrawlSite("http://www.drudgereportfeed.com", "//span[@class='story-headline underline']");
        }

        private static void CrawlSite(string url, string node)
        {
            // CustomObjects class
            List<GEN> listOfThreads = new List<GEN>();

            Console.WriteLine($"1. Crawling Website: {url}");

            // Crawl the Site
            HtmlWeb site = new HtmlWeb();
            HtmlDocument htmlDocument = site.Load(@url);

            // Find all components
            HtmlNodeCollection threads_0 = htmlDocument.DocumentNode.SelectNodes(node);

            Console.WriteLine("Crawled!\n\r");
            Console.WriteLine("2. Constructing objects.");

            // 1. List of thread titles
            for (int i = 0; i < threads_0.Count(); i++)
            {
                string headline = "";
                string link = "";
                link = threads_0[i].GetAttributeValue("href", "");
                headline = threads_0[i].InnerText.Trim();

                GEN ats = new GEN();
                ats.header = headline;
                ats.link = link;

                listOfThreads.Add(ats);
            }

            //Now we save to file.
            Console.WriteLine("Constructed!\n");
            Console.WriteLine("3.Saving to files.");

            if (SaveToTextFile(listOfThreads, url + " " + DateTime.Now.ToString()))
            {
                Console.WriteLine("Saved! Please check " + @"Results\WebCrawlerResults.txt");
            }
            else
            {
                Console.WriteLine("There was an error saving to the file. Please check and try again.");
            }
        }

        private static bool SaveToTextFile(List<GEN> listOfThreads, string title)
        {
            string fileName = @"Results\WebCrawlerResults.txt";

            try
            {
                using (StreamWriter file = new StreamWriter(fileName, true))
                {
                    //file.WriteLine("---------------------");
                    //file.WriteLine(title);
                    //file.WriteLine("");

                    for (int i = 0; i < listOfThreads.Count; i++)
                    {
                        int counter = i + 1;
                        string str = listOfThreads[i].header;
                        str = new string((from c in str
                                          where char.IsWhiteSpace(c) || char.IsLetter(c)
                                          select c
                               ).ToArray());
                        string text = Regex.Replace(str, @"\s+", " ");
                        file.WriteLine(text.ToLower());
                        //file.WriteLine(counter + ", " + listOfThreads[i].header + ", " + listOfThreads[i].link);
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                return false;
            }

            return true;
        }
    }
}
