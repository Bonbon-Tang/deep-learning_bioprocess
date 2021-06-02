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
