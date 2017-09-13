using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpJinx
{
    public class Ciphers
    {
        public static string Text2Num(string text)
        {
            // Converts each char to a number, a=1 b=2 c=3 etc. Thanks (Deccer)
            var encoded = string.Join(string.Empty, text.Select(a => ((int)a) - 96));

            return encoded;
        }
    }
}
