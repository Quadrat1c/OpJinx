using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using HtmlAgilityPack;

namespace OpJinx.Web
{
    public class Crawler
    {
        public static void Start()
        {
            CrawlATS("http://abovetopsecret.com");
        }

        #region Above Top Secret
        private static void CrawlATS(string url)
        {
            // CustomObjects class
            List<ATS> listOfThreads = new List<ATS>();

            Console.WriteLine($"1. Crawling Website: {url}");

            // Crawl the Site
            HtmlWeb site = new HtmlWeb();
            HtmlDocument htmlDocument = site.Load(@url);

            // Find all components
            HtmlNodeCollection threads_0 = htmlDocument.DocumentNode.SelectNodes("//div[@class='headline']");

            Console.WriteLine("Crawled!\n\r");
            Console.WriteLine("2. Constructing objects.");

            // 1. List of thread titles
            for (int i = 0; i < threads_0.Count(); i++)
            {
                string headline = "";
                string link = "";
                link = threads_0[i].GetAttributeValue("href", "");
                headline = threads_0[i].InnerText.Trim();

                ATS ats = new ATS();
                ats.header = headline;
                ats.link = link;

                listOfThreads.Add(ats);
            }

            //Now we save to file.
            Console.WriteLine("Constructed!\n");
            Console.WriteLine("3.Saving to files.");

            if (SaveToFileATS(listOfThreads, "Above Top Secret " + DateTime.Now.ToString()))
            {
                Console.WriteLine("Saved! Please check " + @"Results\WebCrawlerResults.txt");
            }
            else
            {
                Console.WriteLine("There was an error saving to the file. Please check and try again.");
            }
        }

        private static bool SaveToFileATS(List<ATS> listOfThreads, string title)
        {
            string fileName = @"Results\WebCrawlerResults.txt";

            try
            {
                using (StreamWriter file = new StreamWriter(fileName, true))
                {
                    file.WriteLine("---------------------");
                    file.WriteLine(title);
                    file.WriteLine("");

                    for (int i = 0; i < listOfThreads.Count; i++)
                    {
                        int counter = i + 1;
                        file.WriteLine(counter + ", " + listOfThreads[i].header + ", " + listOfThreads[i].link);
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
        #endregion
    }
}