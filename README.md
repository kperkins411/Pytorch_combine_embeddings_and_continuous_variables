# Pytorch how to combine embeddings and continuous variables
Ref: [Justin Johnsons excellent pytorch introduction ](https://github.com/jcjohnson/pytorch-examples)

Convert categorical variables into embeddings and concatenate them with continuous variables that serve as input to a fully connected neural net.

There are several examples in this project that demonstrate concatenating multiple inputs to feed a fully connected neural network.  The most interesting one is [multiple_inputs_embeddings_cats_batch.py](https://github.com/kperkins411/Pytorch_combine_embeddings_and_continuous_variables/blob/master/multiple_inputs_embeddings_cats_batch.py)  since it concatenates both continuous and categorical variables (via embeddings).

## Model summary for [multiple_inputs_embeddings_cats_batch.py](https://github.com/kperkins411/Pytorch_combine_embeddings_and_continuous_variables/blob/master/multiple_inputs_embeddings_cats_batch.py) <br>

   FC_LayerNet(<br>
      (embeddings): Embedding(7, 5)<br>
      (input_linear): Linear(in_features=18, out_features=300, bias=True)<br>
      (middle_linear): Linear(in_features=300, out_features=300, bias=True)<br>
      (output_linear): Linear(in_features=300, out_features=1, bias=True)<br>
   )<br>
   
The first linear layer has 18 inputs consisting of:
embedding(5) + X1(6) + X2(7) = 18 inputs 
