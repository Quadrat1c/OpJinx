using System;
using System.Drawing;
using System.Drawing.Imaging;


namespace JinxNeuralNetwork
{
    public class UtilsDrawing
    {
        //draw line graph 
        /// <summary>
        /// Draw line graph.
        /// </summary>
        /// <param name="points"></param>
        /// <param name="colors"></param>
        /// <param name="tags"></param>
        /// <param name="xMin"></param>
        /// <param name="xMax"></param>
        /// <param name="yMin"></param>
        /// <param name="yMax"></param>
        /// <returns>Bitmap image of drawn line graph.</returns>
        public static Bitmap DrawLineGraph(float[][][] points, Color[] colors, string[] tags, float xMin, float xMax, float yMin, float yMax)
        {
            Bitmap bmp = new Bitmap(640, 640);

            Graphics g = Graphics.FromImage(bmp);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            g.Clear(Color.White);

            const int BORDER_SIZE = 4,
                      X_PADDING = 18,
                      BOTTOM_PADDING = 24;

            Pen pen = new Pen(Color.FromArgb(unchecked((int)0xFF909090)), 1.0f);
            Font font = new Font("Arial", 14);

            //draw graph
            g.DrawRectangle(pen, X_PADDING, BORDER_SIZE, 640 - (X_PADDING + BORDER_SIZE), 640 - BOTTOM_PADDING);
            g.DrawString("1", font, pen.Brush, 2.0f, 10.0f);
            g.DrawString("0", font, pen.Brush, 2.0f, 640.0f - BOTTOM_PADDING);
            g.DrawString("1", font, pen.Brush, 640.0f - 18.0f, 640.0f - 20.0f);

            pen.Color = Color.Black;
            g.DrawString("Y", font, pen.Brush, 2.0f, 320.0f);
            g.DrawString("X", font, pen.Brush, 320.0f, 640.0f - 20.0f);

            //draw legend tags
            if (tags != null)
            {
                for (int i = 0; i < colors.Length; i++)
                {
                    pen.Color = colors[i];
                    g.DrawString(tags[i], font, pen.Brush, 320.0f - tags[i].Length * 5.0f, 140.0f + i * 30.0f);
                }
            }

            //draw lines
            for (int s = 0; s < points.Length; s++)
            {
                int ns = points[s].Length;
                pen.Color = colors[s];
                pen.Width = 2.0f;

                Point[] ps = new Point[ns];
                for (int i = 0; i < ns; i++)
                {
                    ps[i] = new Point((int)(X_PADDING + ((points[s][i][0] - xMin) / (xMax - xMin)) * (640.0f - (X_PADDING + BORDER_SIZE))),
                                      (int)(BORDER_SIZE + (1.0f - (points[s][i][1] - yMin) / (yMax - yMin)) * (640.0f - BOTTOM_PADDING)));
                }

                g.DrawLines(pen, ps);
            }

            g.Dispose();

            return bmp;
        }

        //draw visualization of neural network
        /// <summary>
        /// Draw visualization of NeuralNetwork.
        /// </summary>
        /// <param name="nn"></param>
        /// <param name="synapseAlpha"></param>
        /// <returns>Bitmap image containing drawn NeuralNetwork.</returns>
        public static Bitmap DrawNeuralNetwork(NeuralNetwork nn, float synapseAlpha)
        {
            int imageWidth = Math.Max((2 + nn.hiddenLayers.Length) * 100, 350),
                calcHeight = Math.Max(nn.maxNumberOfHiddenNeurons, Math.Max(nn.outputLayer.numberOfNeurons, nn.inputLayer.numberOfNeurons)) * 6,
                imageHeight = calcHeight + 50;

            Bitmap bmp = new Bitmap(imageWidth, imageHeight);

            Graphics g = Graphics.FromImage(bmp);

            g.Clear(Color.Black);

            Pen pen = new Pen(Color.White, 1.0f);
            Font font = new Font("Arial", 10);


            g.DrawString("Total # Neurons: " + nn.TotalNumberOfNeurons() + " | Total # Synapses: " + nn.TotalNumberOfSynapses(), font, pen.Brush, 4.0f, imageHeight - 20.0f);

            int hlen = nn.hiddenLayers.Length;

            //draw synapses

            //hidden
            float origX = 4.0f;
            for (int i = 0; i < hlen; i++)
            {
                if (i == 0)
                {
                    pen.Color = Color.CadetBlue;
                }
                else
                {
                    pen.Color = Color.DarkGreen;
                }
                origX += 100.0f;
                drawSynapses(g, pen, origX, 100, calcHeight, i == 0 ? nn.inputLayer.numberOfNeurons : nn.hiddenLayers[i - 1].numberOfNeurons, nn.hiddenLayers[i].numberOfNeurons, synapseAlpha, nn.hiddenConnections[i].weights);
            }

            //output
            pen.Color = Color.Goldenrod;
            origX += 100.0f;
            drawSynapses(g, pen, origX, 100, calcHeight, hlen > 0 ? nn.hiddenLayers[hlen - 1].numberOfNeurons : nn.inputLayer.numberOfNeurons, nn.outputLayer.numberOfNeurons, synapseAlpha, nn.outputConnection.weights);

            //draw neuron layers

            //draw hidden layers
            pen.Color = Color.LawnGreen;
            origX = 4.0f;
            for (int i = 0; i < nn.hiddenLayers.Length; i++)
            {
                origX += 100.0f;
                g.DrawString("(" + nn.hiddenLayers[i].numberOfNeurons + "):" + Utils.GetActivationFunctionName(nn.hiddenLayers[i].activationFunction), font, pen.Brush, origX - 15, 4.0f);
                drawNeurons(g, pen, origX, calcHeight, nn.hiddenLayers[i], nn.hiddenRecurringConnections[i], synapseAlpha);
            }

            //draw input layer
            pen.Color = Color.Turquoise;
            g.DrawString("(" + nn.inputLayer.numberOfNeurons + ")input", font, pen.Brush, 4.0f, 4.0f);
            drawNeurons(g, pen, 4.0f, calcHeight, nn.inputLayer, null, synapseAlpha);

            //draw output layer
            pen.Color = Color.Bisque;
            origX += 100.0f;

            g.DrawString("(" + nn.outputLayer.numberOfNeurons + ")output:" + Utils.GetActivationFunctionName(nn.outputLayer.activationFunction), font, pen.Brush, origX - 15, 4.0f);
            drawNeurons(g, pen, origX, calcHeight, nn.outputLayer, null, synapseAlpha);


            g.Dispose();

            return bmp;
        }

        //draw layer of neurons
        private static void drawNeurons(Graphics g, Pen pen, float origX, int calcHeight, NeuralNetworkLayer layer, NeuralNetworkLayerConnection rlayer, float weightAlpha) // int nNeurons, bool recurring)
        {

            float origY = 27.0f + calcHeight / 2.0f - layer.numberOfNeurons * 3.0f;

            for (int i = 0; i < layer.numberOfNeurons; i++)
            {
                float y = origY + i * 6.0f;
                if (layer.recurring)
                {
                    Color old = pen.Color;
                    pen.Color = Color.Maroon;
                    drawRecurringSynapses(g, pen, origX, calcHeight, layer.numberOfNeurons, weightAlpha, rlayer.weights);
                    pen.Color = old;
                }
            }

            for (int i = 0; i < layer.numberOfNeurons; i++)
            {
                float y = origY + i * 6.0f;
                g.DrawRectangle(pen, origX, y, 4.0f, 4.0f);
            }
        }
        //draw array of recurring synapses
        private static void drawRecurringSynapses(Graphics g, Pen pen, float origX, int calcHeight, int n1, float weightAlphaScale, float[] weights)
        {
            float n1OrigY = 29.0f + calcHeight / 2.0f - n1 * 3.0f;

            int sx = (int)(origX + 2.0f);
            Color pcolor = pen.Color;
            int cr = pcolor.R,
                  cg = pcolor.G,
                  cb = pcolor.B;

            int i = n1,
                weightIndex = 0;
            while (i-- > 0)
            {
                int n2y = (int)(n1OrigY + i * 6.0f);

                int k = n1;
                while (k-- > 0)
                {
                    float alpha = Math.Abs(weights[weightIndex++]);
                    if (alpha < 0.0f) alpha = 0.0f;
                    if (alpha > 1.0f) alpha = 1.0f;
                    alpha *= weightAlphaScale;
                    pen.Color = Color.FromArgb((int)(alpha * 255), cr, cg, cb);

                    int nextY = (int)(n1OrigY + k * 6.0f);
                    g.DrawCurve(pen, new Point[] { new Point(sx, n2y), new Point(sx - (int)(-10 + Utils.NextFloat01() * 20), (int)Utils.Lerp(n2y, nextY, Utils.NextFloat01())), new Point(sx, nextY) });
                }
            }

            pen.Color = pcolor;
        }

        //draw array of synapses
        private static void drawSynapses(Graphics g, Pen pen, float origX, int shiftX, int calcHeight, int n1, int n2, float weightAlphaScale, float[] weights)
        {
            float n1OrigY = 29.0f + calcHeight / 2.0f - n1 * 3.0f,
                  n2OrigY = 29.0f + calcHeight / 2.0f - n2 * 3.0f;

            float sx = origX + 2.0f - shiftX;
            Color pcolor = pen.Color;
            int cr = pcolor.R,
                  cg = pcolor.G,
                  cb = pcolor.B;

            float ox = origX + 2.0f;

            int i = n2,
                weightIndex = 0;
            while (i-- > 0)
            {
                float n2y = n2OrigY + i * 6.0f;

                int k = n1;
                while (k-- > 0)
                {
                    float alpha = Math.Abs(weights[weightIndex++]);
                    if (alpha < 0.0f) alpha = 0.0f;
                    if (alpha > 1.0f) alpha = 1.0f;
                    alpha *= weightAlphaScale;
                    pen.Color = Color.FromArgb((int)(alpha * 255), cr, cg, cb);

                    g.DrawLine(pen, ox, n2y, sx, n1OrigY + k * 6.0f);
                }
            }

            pen.Color = pcolor;
        }


        /// <summary>
        /// Sample neuralnetwork as 2D function, building image.
        /// </summary>
        /// <param name="nn"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public static Bitmap AsImage(NeuralNetwork nn, int width, int height)
        {
            NeuralNetworkProgram nnp = new NeuralNetworkProgram(nn);
            nnp.context.Reset(true);

            float[] id = nnp.context.inputData,
                od = nnp.context.outputData;

            Bitmap bmp = new Bitmap(width, height);
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    id[0] = x / (float)width;
                    id[1] = y / (float)height;
                    nnp.Execute();
                    bmp.SetPixel(x, y, Color.FromArgb((int)(od[0] * 255), (int)(od[1] * 255), (int)(od[2] * 255)));
                }
            }
            return bmp;
        }


        /// <summary>
        /// Convert 'area' of Bitmap 'bmp' to RGB float data, for neural network input. 
        /// </summary>
        /// <param name="bmp"></param>
        /// <param name="area"></param>
        /// <returns></returns>
        public static float[] ToRGBData(Bitmap bmp, Rectangle area)
        {
            float[] d = new float[area.Width*area.Height*3];

            BitmapData bdat = bmp.LockBits(area, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            unsafe
            {
                byte* px = (byte*)bdat.Scan0.ToPointer();
                int stride = bdat.Stride,
                    ystride = area.Width*3;

                for (int x = 0; x < area.Width; x++)
                {
                    for (int y = 0; y < area.Height; y++)
                    {
                        int pxInd = x * 3 + y * stride,
                            dind = x * 3 + y * ystride;

                        d[dind] = px[pxInd] / 255.0f;
                        dind++;
                        pxInd++;
                        d[dind] = px[pxInd] / 255.0f;
                        dind++;
                        pxInd++;
                        d[dind] = px[pxInd] / 255.0f;
                    }
                }
            }
            bmp.UnlockBits(bdat);

            return d;
        }


        /// <summary>
        /// Convert Bitmap 'bmp' into chunks of RGB float data of desired 'chunkSize', for neural network input. 
        /// </summary>
        /// <param name="bmp"></param>
        /// <param name="chunkSize"></param>
        /// <returns></returns>
        public static float[][] ToRGBData(Bitmap bmp, int chunkSize)
        {
            int nx = bmp.Width/chunkSize - 1,
                ny = bmp.Height/chunkSize - 1;

            float[][] rd = new float[nx * ny][];
            for (int x = 0; x < nx; x++)
            {
                for (int y = 0; y < ny; y++)
                {
                    rd[x + y * nx] = ToRGBData(bmp, new Rectangle(x * chunkSize, y * chunkSize, chunkSize, chunkSize));
                }
            }

            return rd;
        }


        /// <summary>
        /// Fill Bitmap 'bmp' with data from RGB data 'd'.
        /// </summary>
        /// <param name="bmp"></param>
        /// <param name="d"></param>
        public static void FromRGBData(Bitmap bmp, float[] d)
        {
            BitmapData bdat = bmp.LockBits(new Rectangle(0,0,bmp.Width,bmp.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            unsafe
            {
                byte* px = (byte*)bdat.Scan0.ToPointer();
                int stride = bdat.Stride,
                    ystride = bmp.Width * 3;

                for (int x = 0; x < bmp.Width; x++)
                {
                    for (int y = 0; y < bmp.Height; y++)
                    {
                        int pxInd = x * 3 + y * stride,
                            dind = x * 3 + y * ystride;

                        px[pxInd] = (byte)(d[dind] * 255);
                        dind++;
                        pxInd++;
                        px[pxInd] = (byte)(d[dind] * 255);
                        dind++;
                        pxInd++;
                        px[pxInd] = (byte)(d[dind] * 255);
                    }
                }
            }
            bmp.UnlockBits(bdat);
        }
    }
}
