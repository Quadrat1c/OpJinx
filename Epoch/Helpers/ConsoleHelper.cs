using System;

namespace Epoch
{
    public class ConsoleHelpers
    {
        public static void Intro()
        {
            Console.ForegroundColor = ConsoleColor.DarkCyan;
            Console.WriteLine(@"___________                    .__     ");
            Console.WriteLine(@"\_   _____/_____   ____   ____ |  |__  ");
            Console.WriteLine(@" |    __)_\____ \ /  _ \_/ ___\|  |  \ ");
            Console.WriteLine(@" |        \  |_> >  <_> )  \___|   Y  \");
            Console.WriteLine(@"/_______  /   __/ \____/ \___  >___|  /");
            Console.WriteLine(@"        \/|__|               \/     \/ ");
            Console.ForegroundColor = ConsoleColor.White;
        }

        public static string GetLine()
        {
            var line = Console.ReadLine();
            return line?.Trim() ?? string.Empty;
        }

        public static int GetInput(string message, int min)
        {
            Console.Write(message);
            var num = GetNumber();

            while (num < min)
            {
                Console.Write(message);
                num = GetNumber();
            }

            return num;
        }

        public static int GetNumber()
        {
            int num;
            var line = GetLine();
            return line != null && int.TryParse(line, out num) ? num : 0;
        }

        public static bool GetBool(string message)
        {
            Console.WriteLine(message);
            Console.Write("Answer: ");
            var line = GetLine();

            bool answer;
            while (line == null || !TryGetBoolResponse(line.ToLower(), out answer))
            {
                if (line == "exit")
                    Environment.Exit(0);

                Console.WriteLine(message);
                Console.Write("Answer: ");
                line = GetLine();
            }

            PrintNewLine();
            return answer;
        }

        public static bool TryGetBoolResponse(string line, out bool answer)
        {
            answer = false;
            if (string.IsNullOrEmpty(line)) return false;

            if (bool.TryParse(line, out answer)) return true;

            switch (line[0])
            {
                case 'y':
                    answer = true;
                    return true;
                case 'n':
                    return true;
            }

            return false;
        }

        public static void PrintNewLine(int numNewLines = 1)
        {
            for (var i = 0; i < numNewLines; i++)
                Console.WriteLine();
        }

        public static void PrintUnderline(int numUnderlines)
        {
            for (var i = 0; i < numUnderlines; i++)
                Console.Write('-');
            PrintNewLine(2);
        }

        public static void WriteError(string error)
        {
            Console.WriteLine(error);
            Console.ReadLine();
            Environment.Exit(0);
        }
    }
}
