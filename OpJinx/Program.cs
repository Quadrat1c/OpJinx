using System;

namespace OpJinx
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Operation Jinx");

            Words.DensityCheck("http://abovetopsecret.com");
            Console.ReadLine();

            Words.DensityCheck("http://godlikeproductions.com");
            Console.ReadLine();

            Words.DensityCheck("http://infowars.com");
            Console.ReadLine();

            Words.DensityCheck("http://cnn.com");

            Console.Write("Press any key to exit.");
            Console.ReadLine();
        }
    }
}
