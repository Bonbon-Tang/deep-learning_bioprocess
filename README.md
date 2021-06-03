# deep-learning_bioprocess
This is a mini-project that uses deep-learning methods to model a bioprocess with/without noise.

  
The first attempt to model the bioprocess is based on the noise-free data. This was generated via given math-equations and ODE-solver in python. 
When training data, the given math equations were assumed to be unknown;instead, data was the only resource which can be used for buidling model.

#########################################################################

Noise-free Model

#########################################################################

The first attempt to model the bioprocess is based on the noise-free data. This was generated via given math-equations and ODE-solver in python. 
When training data, the given math equations were assumed to be unknown; instead, data was the only resource which can be used for buidling model.

Firstly, five set of data was generated via the method above. The names were output*_*_*_t, the "*" indicates the initial condition of the three measured points.
The time-span of this data was 1000s, with 0.5s being the measurement interval. 

The data generated could not be used directly as the magnitute of measurement was largely different. 
Thus, scaling is evitable and here, constant "mean" and "variance" was used, these two file was also saved in packages. 

As for training,validation and testing, two-set(1,150,0)(0.9,140,0),one-set(0.95,145,0) were allocated as training-set and validation-set respectively. The rest were testing set.
The purposed model are series-of-Feed-forward neural nework and GRU:

The first model was three Feed-Forward neural networks with 0.5s,2s,8s intervals. All of them had only one-hidden layer with 20 hidden units. The inputs and outputs were 3.
For testing, they were conmbined for prediction. Given that every 32s, a ground-true point were known and the next points need to be predicted via models above. 
The first testing shows that the prediction was perfect while the second testing was not that perfect. This is understandable because the second testing were far from the
region of the training set. This was done to show how the model decays when the testing were far from the training. 

The second model was a basic GRU model. Here, teacher training was used to train and this will reduce the training time exponentially. 
When predicing, the model was expected to predict next 300-time step values. It should be mentioned that the input could be 1-step or several time-steps, this will be 
sent to encoder first and the encoder_output will be then sent to decoder for step-to-step(300steps expected) prediction. 
The testing data within the range of training set shows a very good result, but that far from the training set will deviate after hundreds of time-steps.

#########################################################################

Modelling with noise

#########################################################################

  
Similar to modelling the bioprocess without noise, data was generated via equations and ODE solvers. Gaussian white noise was added to the ground truth value.
This time, more sets of data were generated as noisy data was much more difficult to model. A total of 6 sets of data was generated and half of them were used for training. 

Here, three models was proposed. Series of FNN and GRU were similar to those in noise-free modelling while the last was a transformer model. 
Before training models, the data needed to be preprocessed as previous. They were standarized first.
The obtained mean and variance was used both for training and testing. (Saved in file)

The first model proposed was series of FNN. A total of 3 FNN was trained(1,4,16steps) and was used to predict next 300 steps given the first step. 
All of the separated FNN has only one hidden layer with 20 hidden units.
As can be seen in the testing diagram, although the model will deviate somewhat after hundreds of timesteps, it can still capture the main trend of the bioprocess.
Compared to the model for noise-free data, it was somewhat noisy and inaccurate; however, note that there are still only 3 FNN, if more are trained, this might be alleviated.

The second model was GRU. The number hidden units was also 20. Like the one in noise-free model, a encoder-decoder model was proposed. An arbitrary number of known points 
can be sent into the encoder, the decoder will output the next several steps( can be determined ).
As in the testing diagram, the problem is similar to that in SFNN but less. Long-time prediction was not that perfect but the trend was much more smooth than that of SFNN.

Finally, a tranformer-model was used. It was exactly the model proposed in the original paper "attention is all you need" with encoder and decoder. The encoder here takes the
known points and sends the encoder-result to decoder for prediction. The positional ecoding was used to enhance the time-information and masks were also used for prediction.
It should also be menstioned that teacher-training method was also adopted here to accelerate the training- process. Actually, the training of Transformer was much faster than
GRU-training due to paralism. 
The result of prediction is even slightly better than those of GRU-model(shown in the diagram), although the curves are non-smooth. 

In conclusion, although series of FNN perfomed extremely well for noise-free modelling, it was not that ideal for noisy modelling. By conparison, GRU and transformer are more
likely to catch the long-time depedency of data. GRU can predict the curve more smoothly while transformer seems to be more robust for noise.
