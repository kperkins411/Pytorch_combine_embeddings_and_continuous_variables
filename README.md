# Pytorch how to combine embeddings and continuous variables
Convert categorical variables into embeddings and concat them with with continuous variables that serve as input to a fully connected neural net.

Ref: [Justin Johnsons excellent pytorch introduction ](https://github.com/jcjohnson/pytorch-examples)

This project builds on the above project using examples.  The most interesting one is [  ] (embeddings) since it concatenates both continuous and categorical variables (via embeddings) into an input for a fully connected neural network.  

This is a summary of the model created.  
FC_LayerNet(
  (embeddings): Embedding(7, 5)
  (input_linear): Linear(in_features=18, out_features=300, bias=True)
  (middle_linear): Linear(in_features=300, out_features=300, bias=True)
  (output_linear): Linear(in_features=300, out_features=1, bias=True)
)
The first linear layer has 18 inputs consisting of:
embedding(5) + X1(6) + X2(7) = 18 inputs 
