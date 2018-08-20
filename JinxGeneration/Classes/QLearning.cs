using System;
using System.Threading;
using System.IO;
using System.Collections.Generic;

namespace JinxNeuralNetwork
{
    /// <summary>
    /// Trains a NeuralNetwork through QLearning(action-reward system)
    /// </summary>
    public class NeuralNetworkQLearning
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="actionId"></param>
        public delegate void OnLearningReplayAction(int actionId);

        /// <summary>
        /// 
        /// </summary>
        public OnLearningReplayAction onReplayAction = null;



        private bool hasRecurring;

        private NeuralNetwork neuralNetwork;


		private string learningSessionsFile = null;
		private Stream learningSessionStream = null;
		private int currentSession = 0;
		private bool beganSession = false;
		private List<int> sessions = new List<int>();

        private NeuralNetworkContext[] stackedRuntimeContext;
        private NeuralNetworkFullContext[] stackedFullContext;
        private NeuralNetworkPropagationState[] stackedDerivativeMemory;
        private NeuralNetworkDerivativeMemory derivatives = new NeuralNetworkDerivativeMemory();

        private int maxUnrollLength;


        /// <summary>
        /// Create new NeuralNetworkQLearning system. 
        /// </summary>
        /// <param name="nn">NeuralNetwork to train.</param>
        /// <param name="inputDat">Input data.</param>
        /// <param name="targetDat">Target data.</param>
		public NeuralNetworkQLearning(NeuralNetwork nn, int maxUnrollLen, string sessionsFileName)
        {
            neuralNetwork = nn;
			learningSessionsFile = sessionsFileName;

            maxUnrollLength = maxUnrollLen;
            if (maxUnrollLength < 1) maxUnrollLength = 1;

            //check for recurring layer, if need to stack and unroll
            for (int i = 0; i < nn.hiddenLayers.Length; i++)
            {
                if (nn.hiddenLayers[i].recurring)
                {
                    hasRecurring = true;
                    break;
                }
            }
            

            derivatives.Setup(nn);
            if (!hasRecurring)
            {
                maxUnrollLength = 1;
            }

            stackedRuntimeContext = new NeuralNetworkContext[maxUnrollLength];
            stackedFullContext = new NeuralNetworkFullContext[maxUnrollLength];
            stackedDerivativeMemory = new NeuralNetworkPropagationState[maxUnrollLength];
            for (int i = 0; i < stackedRuntimeContext.Length; i++)
            {
                stackedRuntimeContext[i] = new NeuralNetworkContext();
                stackedRuntimeContext[i].Setup(nn);

                stackedFullContext[i] = new NeuralNetworkFullContext();
                stackedFullContext[i].Setup(nn);

                stackedDerivativeMemory[i] = new NeuralNetworkPropagationState();
                stackedDerivativeMemory[i].Setup(nn, stackedRuntimeContext[i], stackedFullContext[i], derivatives);
            }
			/*
            if (hasRecurring)
            {
                recurringMemoryState = new float[nn.hiddenLayers.Length][];
                for (int i = 0; i < nn.hiddenLayers.Length; i++)
                {
                    if (nn.hiddenLayers[i].recurring)
                    {
                        recurringMemoryState[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                    }
                }
            }*/
        }
		~NeuralNetworkQLearning() {
			if (learningSessionStream != null && (learningSessionStream.CanRead || learningSessionStream.CanWrite))
				learningSessionStream.Close ();
		}

        /// <summary>
        /// Begins new learning session.
        /// </summary>
        public void Start()
        {
			learningSessionStream = File.OpenWrite (learningSessionsFile);
			currentSession = 0;

			SaveRecurringMemoryState ();
		}

		/// <summary>
		/// Restarts session.
		/// </summary>
		public void RestartSession() {
			ClearSession ();

			currentSession = 0;
			SaveRecurringMemoryState ();
		}

		/// <summary>
		/// Clears current session.
		/// </summary>
		public void ClearSession() {
			if (beganSession) {
				long recurrMemorySz = 0;
				float[][] hm = stackedRuntimeContext [0].hiddenRecurringData;
				for (int i = 0; i < hm.Length; i++) {
					if (hm [i] != null) {
						recurrMemorySz += hm [i].Length * 4;
					}
				}
				long len = learningSessionStream.Length - (currentSession * (long)(4 + neuralNetwork.inputLayer.numberOfNeurons * 4) + recurrMemorySz);
				learningSessionStream.Seek(len-1, SeekOrigin.Begin);
				learningSessionStream.SetLength (len);

				beganSession = false;
			}
		}

		/// <summary>
		/// Clears all saved learning sessions, you must manually call RestartSession after this.
		/// </summary>
		public void ClearAllSessions() {
			File.WriteAllBytes (learningSessionsFile, new byte[0]);
		}


		/// <summary>
		/// Save QLearning learning session state to stream.
		/// </summary>
		/// <param name="s">S.</param>
		public void Save(Stream s) {
			Utils.IntToStream (sessions.Count, s);
			for (int i = 0; i < sessions.Count; i++)
				Utils.IntToStream (sessions [i], s);
		}

		/// <summary>
		/// Load QLearning learning session state from stream.
		/// </summary>
		/// <param name="s">S.</param>
		public void Load(Stream s) {
			sessions.Clear ();
				int nsession = Utils.IntFromStream (s);
			for (int i = 0; i < nsession; i++)
				sessions.Add(Utils.IntFromStream (s));
		}


        //copies c1 recurring data to c2
        private void CopyRecurringState(NeuralNetworkContext c1, NeuralNetworkContext c2)
        {
            int i = c1.hiddenRecurringData.Length;
            while (i-- > 0)
            {
                if (c1.hiddenRecurringData[i] == null) continue;
                Array.Copy(c1.hiddenRecurringData[i], c2.hiddenRecurringData[i], c1.hiddenRecurringData[i].Length);
            }
        }


        /// <summary>
        /// 
        /// </summary>
        private void SaveRecurringMemoryState()
        {
			float[][] hm = stackedRuntimeContext[0].hiddenRecurringData;
			for (int i = 0; i < hm.Length; i++)
            {
                if (hm[i] != null)
                {
					Utils.FloatArrayToStream (hm [i], learningSessionStream);
                }
            }
			beganSession = true;
		}



        /// <summary>
        /// 
        /// </summary>
        /// <returns>ID of selected action neuron.</returns>
        public int Execute()
        {
            NeuralNetworkContext ctx = stackedRuntimeContext[0];
            neuralNetwork.Execute(ctx);

            //randomly select action from output result
            Utils.Normalize(ctx.outputData);
            int action = Utils.RandomChoice(ctx.outputData);

			//save action to session
			Utils.IntToStream (action, learningSessionStream);
			Utils.FloatArrayToStream (ctx.inputData, learningSessionStream);
			currentSession++;

            return action;
        }


        /// <summary>
        /// Finishes session with reward and starts another new session. Call 'Learn' to iterate through sessions training.
        /// </summary>
        /// <param name="amount"></param>
        public void Reward(float reward)
		{
			Utils.FloatToStream (reward, learningSessionStream);
			sessions.Add (currentSession);
			RestartSession ();
		}

		/// <summary>
		/// Learn from rewarded sessions with specified 'learningRate', 'iter' times.
		/// </summary>
		/// <param name="learningRate">Learning rate.</param>
		public void Learn(float learningRate, int iter) {
			//clear current session from stream and end session stream
			ClearSession ();
			learningSessionStream.Close ();

			//begin reading session stream
			learningSessionStream = File.OpenRead (learningSessionsFile);


			float[] tb = stackedRuntimeContext [0].outputData;
			float[][] hm = stackedRuntimeContext[0].hiddenRecurringData;
			QLearningContext[] qctx = new QLearningContext[maxUnrollLength];
			for (int i = 0; i < maxUnrollLength; i++) {
				qctx[i] = new QLearningContext (0, new float[neuralNetwork.inputLayer.numberOfNeurons]);
			}

			for (int j = 0; j < iter; j++) {
				learningSessionStream.Position = 0;

				//training
				for (int s = 0; s < sessions.Count; s++) {
					//reset derivatives/context memory
					derivatives.Reset ();
					for (int i = 0; i < maxUnrollLength; i++) {
						stackedRuntimeContext [i].Reset (true);
						stackedDerivativeMemory [i].Reset ();
					}

					//initial memory state
					for (int i = 0; i < hm.Length; i++)
					{
						if (hm[i] != null)
						{
							Utils.FloatArrayFromStream (hm [i], learningSessionStream);
						}
					}


					int alen = sessions[s],
					unrollCount = 0;

					//seek ahead to load reward then back
					long lpos = learningSessionStream.Position;
					learningSessionStream.Seek (lpos + alen*(4+neuralNetwork.inputLayer.numberOfNeurons*4), SeekOrigin.Begin);
					float rewardAmount = Utils.FloatFromStream (learningSessionStream)*learningRate;
					learningSessionStream.Seek (lpos, SeekOrigin.Begin);

					for (int i = 0; i < alen; i++) {
						qctx[unrollCount].action = Utils.IntFromStream (learningSessionStream);
						Utils.FloatArrayFromStream (qctx[unrollCount].input, learningSessionStream);

						Array.Copy (qctx[unrollCount].input, stackedRuntimeContext [unrollCount].inputData, qctx[unrollCount].input.Length);
						neuralNetwork.Execute_FullContext (stackedRuntimeContext [unrollCount], stackedFullContext [unrollCount]);
						if (onReplayAction != null)
							onReplayAction (qctx[unrollCount].action);
						

						unrollCount++;
						if (unrollCount >= maxUnrollLength || i + 1 >= alen) {
							//back propagate through stacked
							int tdatIndex = i;
							while (unrollCount-- > 0) {
								tb [qctx[unrollCount].action] = 1.0f;
								neuralNetwork.ExecuteBackwards (tb, stackedRuntimeContext [unrollCount], stackedFullContext [unrollCount], stackedDerivativeMemory [unrollCount], NeuralNetworkTrainer.LOSS_TYPE_AVERAGE, -1);
								tb [qctx[unrollCount].action] = 0.0f;

								tdatIndex--;
							}

							//learn
							NeuralNetworkAdaGradMemory.ApplyNoMemory (stackedDerivativeMemory [0], stackedDerivativeMemory[0].weights, stackedDerivativeMemory[0].biases, stackedDerivativeMemory[0].recurrWeights, rewardAmount);
							derivatives.Reset ();
							unrollCount = 0;

							if (i + 1 >= alen) {
								//not enough room for another full length propagation
							} else {
								//copy recurring state over
								CopyRecurringState (stackedRuntimeContext [maxUnrollLength - 1], stackedRuntimeContext [0]);
							}
						} else {
							//copy recurring state into next
							CopyRecurringState (stackedRuntimeContext [unrollCount - 1], stackedRuntimeContext [unrollCount]);
						}
					}
				}
			}

        }

        /// <summary>
        /// Get neural network runtime context.
        /// </summary>
        /// <returns></returns>
        public NeuralNetworkContext GetNeuralNetworkContext()
        {
            return stackedRuntimeContext[0];
        }
    }


    public class QLearningContext
    {
        public int action;
        public float[] input;

        public QLearningContext(int a, float[] i)
        {
            action = a;
            input = new float[i.Length];
            Array.Copy(i, input, i.Length);
        }
    }
}
