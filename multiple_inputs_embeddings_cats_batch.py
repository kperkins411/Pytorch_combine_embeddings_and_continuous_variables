#!/usr/bin/env python

'''
How to combine categorical data (embeddings) with continuous data in a pytorch fully connected neural net
'''

import torch
import numpy as np

#repeatability
torch.manual_seed(1)

#run on gpu if possible, otherwise run on CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu") # for test cause its faster to load

#parameters
x1_dim          = 6    #input width
x2_dim          = 7    #"
embedding_dim   = 5    #width of each embedding row
N               = 64   #batch size
D_in = x1_dim + x2_dim + embedding_dim  #width of input to first hidden layer
D_out           = 1    #width of output
H1              = 300  #width of hiden layer
EPOCHS          = 200

class FC_LayerNet(torch.nn.Module):
    expected_layer_sizes = 3
    def __init__(self, layer_szs, embedding_szs,  embedding_num, embedding_dim):
        '''
        how to combine a bunch of inputs and feed into a neural net
        :param layer_szs: list of layer sizes, 1st entry is the input size, last is output size
        :param embedding_szs: list of embedding tuples, each tuple of type (num_unique_embedding, width_each_embedding)

        :param D_in: width of input, in this case its the concatenation of the embedding with both x1_dim and x2_dim
        :param H: number neurons in hidden layer
        :param D_out: width of output
        :param embedding_num: number of embeddings
        :param embedding_dim: width of each
        '''
        super(FC_LayerNet, self).__init__()

        assert len(layer_sizes) == FC_LayerNet.expected_layer_sizes

        self.embeddings = torch.nn.Embedding(embedding_num, embedding_dim)

        d_in = layer_sizes[0]
        d_H = layer_sizes[1]
        d_out = layer_sizes[2]

        #7 inputs because x,y and 5 embeddings
        self.input_linear  = torch.nn.Linear(d_in, d_H)
        self.middle_linear = torch.nn.Linear(d_H,  d_H)
        self.output_linear = torch.nn.Linear(d_H,  d_out)


    def forward(self, x,y,day):
        '''
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        :param x: N sized batch of data
        :param y: N sized batch of different data
        :param day: N sized batch of days
        :return: N sized tensor of predictions
        '''
        #map the integer day to the row associated with that day in the embedding matrix
        embeds = self.embeddings(day)

        # concat x, y and embeds, if x is 5 wide, y is 6 wide and embeds is 7 wide, then xce will be 18 wide
        xc = torch.cat((x,y),dim = 1)
        xce = torch.cat((xc,embeds),dim = 1)

        h_relu = self.input_linear(xce).clamp(min=0)  #run through linear layer1 and then relu
        h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

#create continuous inputs
x1 = torch.randn(N, x1_dim, device=device)
x2 = torch.randn(N, x2_dim, device=device)

#create categorical input
days = ["Sun","mon","tues","wed","thurs","fri","sat"]
# days = list(set(days) ) #unneeded here
num_days = len(days)  #how many in category?
days_to_idx = {day:i for i, day in enumerate(days)}                 #dict day to index

cat_days =  [ days[np.random.randint(0,7)] for j in range(0,N)]     #N long list of random days
cat_days_index = [days_to_idx[day] for day in cat_days]             #N long list of intergers from days_to_idx[day]

#create expected output
y = torch.randn(N, D_out, device=device)

# the model

layer_sizes = [D_in, H1, D_out]
model = FC_LayerNet(layer_sizes, embedding_num = num_days,embedding_dim = embedding_dim )
if (device.type == 'cuda'):
    model.cuda()

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(EPOCHS):
    total_loss = torch.Tensor([0])
    # Prepare the inputs to be passed to the model (i.e, turn the days
    # into integer indices and wrap them in variables)

    # day = cat_days[j]
    context_idx = torch.tensor([days_to_idx[day] for day in cat_days], dtype=torch.long, device = device)

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x1,x2,context_idx)

    # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
    # is a Python number giving its value.
    y_1 = torch.tensor(y, dtype=torch.float, device = device)
    loss = loss_fn(y_pred,y_1)

    # Get the Python number from a 1-element Tensor by calling tensor.item()
    total_loss += loss.item()
    # print(model.embeddings._parameters['weight'][0][:5])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch %s, loss =%s"%(epoch, str(total_loss.item())))


