# Pytorch how to combine embeddings and continuous variables
Convert categorical variables into embeddings and concat them with with continuous variables that serve as input to a fully connected neural net.

Ref: [Justin Johnsons excellent pytorch introduction ](https://github.com/jcjohnson/pytorch-examples)

This project builds on the above project using examples.  The most interesting one is [multiple_inputs_embeddings_cats_batch.py](https://github.com/kperkins411/Pytorch_combine_embeddings_and_continuous_variables/blob/master/multiple_inputs_embeddings_cats_batch.py)  since it concatenates both continuous and categorical variables (via embeddings) into an input for a fully connected neural network.  

## Model summary 
for [multiple_inputs_embeddings_cats_batch.py](https://github.com/kperkins411/Pytorch_combine_embeddings_and_continuous_variables/blob/master/multiple_inputs_embeddings_cats_batch.py) <br>

FC_LayerNet(<br>
   (embeddings): Embedding(7, 5)<br>
  (input_linear): Linear(in_features=18, out_features=300, bias=True)<br>
  (middle_linear): Linear(in_features=300, out_features=300, bias=True)<br>
  (output_linear): Linear(in_features=300, out_features=1, bias=True)<br>
)<br>
The first linear layer has 18 inputs consisting of:
embedding(5) + X1(6) + X2(7) = 18 inputs 
