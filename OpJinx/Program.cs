using System;

namespace OpJinx
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Operation Jinx");

            Words.wordDensity("http://abovetopsecret.com");
            Console.ReadLine();

            Words.wordDensity("http://godlikeproductions.com");
            Console.ReadLine();

            Words.wordDensity("http://infowars.com");
            Console.ReadLine();

            Words.wordDensity("http://cnn.com");

            Console.Write("Press any key to exit.");
            Console.ReadLine();
        }
    }
}
