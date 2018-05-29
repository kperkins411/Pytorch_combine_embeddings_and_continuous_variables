import torch
import numpy as np
torch.manual_seed(1)
np.random.seed(1)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out, embedding_num, embedding_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.embeddings = torch.nn.Embedding(embedding_num, embedding_dim)

        #7 inputs because x,y and 5 embeddings
        self.linear1 = torch.nn.Linear(7, H)
        self.linear2 = torch.nn.Linear(H, D_out)


    def forward(self, x,y,day):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        #get embeds
        embeds = self.embeddings(day).view((1, -1)).squeeze()

        # combine x, y and embeds
        xc = torch.cat((x,y),0)
        xce = torch.cat((xc,embeds),0)

        h_relu = self.linear1(xce).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# dtype = torch.float32
device = torch.device("cpu")
# torch.cuda.current_device()
# dtype = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in1,D_in2, H, D_out = 64, 600,300, 100, 1

# Create random input and output data
x1 = [np.random.randn() for i in range(N)]
x2 = [np.random.randn() for i in range(N)]
y = [np.random.randn() for i in range(N)]

# create some categorical data, days
days = ["Sun","mon","tues","wed","thurs","fri","sat"]
days = list(set(days) )   #how many in category?
num_days = len(days)

cat_days =  [ days[np.random.randint(0,7)] for j in range(0,N)]

#create lookup table
days_to_idx = {day:i for i, day in enumerate(days)}

#convert cat_days to associated indices in days_to_idx
cat_days_index = [days_to_idx[day] for day in cat_days]

model = TwoLayerNet(D_in1,H, D_out, embedding_num = num_days,embedding_dim = 5 )

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 4e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1000):
    total_loss = torch.Tensor([0])
    for j in range(N):
        # Prepare the inputs to be passed to the model (i.e, turn the days
        # into integer indices and wrap them in variables)

        day = cat_days[j]
        context_idx = torch.tensor([days_to_idx[day]], dtype=torch.long)
        x_1 = torch.tensor(x1[j]).unsqueeze(0)
        x_2 = torch.tensor(x2[j]).unsqueeze(0)

        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x_1,x_2,context_idx)

        # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
        # is a Python number giving its value.
        y_1 = torch.tensor(y[j], dtype=torch.float)
        loss = loss_fn(y_pred,y_1)

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
        # print(model.embeddings._parameters['weight'][0][:5])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch %s, loss =%s"%(epoch, str(total_loss)))


