using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text.RegularExpressions;

namespace OpJinx
{
    public class Words
    {
        public static bool debug = false;

        public static void DensityCheck(string site)
        {
            using (WebClient client = new WebClient())
            {
                // Convert webpage to a string
                string html = client.DownloadString(site).ToLower();
                #region Debug
                if (debug == true)
                {
                    Console.WriteLine("------------------------------ html to lower ------------------------------");
                    Console.WriteLine(html);
                }
                #endregion

                html = RemoveHtmlTags(html);
                #region Debug
                if (debug == true)
                {
                    Console.WriteLine("------------------------------ html tags removed ------------------------------");
                    Console.WriteLine(html);
                }
                #endregion

                // Split the string by spaces
                List<string> list = html.Split(' ').ToList();

                // Remove non alphabet words
                var onlyAlphabetRegEx = new Regex(@"^[A-z]+$");
                list = list.Where(f => onlyAlphabetRegEx.IsMatch(f)).ToList();
                #region Debug
                if (debug == true)
                {
                    Console.WriteLine("------------------------------ alphabet words ------------------------------");
                    Console.WriteLine(list);
                }
                #endregion

                // Blacklist words and any word under 2 characters
                string[] blacklist = { "a", "an", "on", "of", "or", "as", "i", "in", "is", "to", "the", "and", "for", "with", "not", "by",
                    "alex", "jones", "above", "top", "secret", "coward", "sep", "rss", "that", "was", "ats", "some", "about", "you", "who",
                    "are", "have", "new", "from", "what", "our", "she", "all", "here", "content", "down", "did", "can", "find", "come",
                    "just", "know", "but", "now", "one", "topics", "how", "will", "there", "its", "news", "they", "over", "this", "has",
                    "like", "your", "while", "says", "before", "still", "today", "anonymous", "should", "why", "world", "show", "infowars",
                    "store", "special", "reports", "view", "after", "cnn", "edition", "set", "page", "glp", "guy", "get", "his", "only",
                    "been", "next", "use", "most", "tells", "david", "knight", "watch", "video", "confirm" };

                list = list.Where(x => x.Length > 2).Where(x => !blacklist.Contains(x)).ToList();
                #region Debug
                if (debug == true)
                {
                    Console.WriteLine("------------------------------ after blacklist ------------------------------");
                    Console.WriteLine(list);
                }
                #endregion
                var keywords = list.GroupBy(x => x).OrderByDescending(x => x.Count());

                foreach (var word in keywords)
                {
                    if (word.Count() > 1)
                        Console.WriteLine($"{word.Key} {word.Count()}");
                }
            }
        }

        static string RemoveHtmlTags(string html)
        {
            string htmlRemoved = Regex.Replace(html, @"<script[^>]*>[\s\S]*?</script>|<[^>]+>|&nbsp;", " ").Trim();
            string normalised = Regex.Replace(htmlRemoved, @"\s{2,}", " ");
            return normalised;
        }
    }
}