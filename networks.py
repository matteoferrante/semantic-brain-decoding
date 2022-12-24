import torch
from torch import nn 
import numpy as np



def get_activation(activation):
    if activation=="relu":
        return nn.ReLU()
    
    elif activation=="gelu":
        return nn.GELU()
    
    elif activation=="tanh":
        return nn.Tanh()
    
    elif activation=="sigmoid":
        return nn.Sigmoid()

    elif activation=="selu":
        return nn.SELU()

class BrainAttentionMLP(nn.Module):
    def __init__(self, latent_dim, hidden=[128], dropout=0.2, activation="gelu", attn_hidden=128):
        """
        Initializes the BrainAttentionMLP model.
        
        Parameters:
        - latent_dim: Dimensionality of the latent space
        - hidden: List of hidden layer sizes for the MLP
        - dropout: Dropout rate for the MLP
        - activation: Activation function for the MLP
        - attn_hidden: Hidden size for the attention layers
        """
        super().__init__()
        
        # Initialize the attention layers
        self.Q = nn.LazyLinear(attn_hidden)
        self.K = nn.LazyLinear(attn_hidden)
        self.V = nn.LazyLinear(attn_hidden)
        self.attention = nn.MultiheadAttention(attn_hidden, num_heads=8)
        
        # Initialize the MLP layers
        self.model = []
        for h in hidden:
            self.model.append(nn.LazyLinear(h))
            self.model.append(get_activation(activation))
        self.model.append(nn.Dropout(dropout))
        self.model.append(nn.LazyLinear(latent_dim))
        # Create a sequential model from the MLP layers
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        - x: Input data
        
        Returns:
        - attn_output: Output of the attention layers
        """
        # Get the query, key, and value for the attention layers
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)
        # Calculate the attention output and attention weights
        attn_output, attn_output_weights = self.attention(query, key, value)
        # Pass the attention output through the MLP
        return self.model(attn_output)




class BrainMLP(nn.Module):
    def __init__(self,latent_dim,hidden=[128],dropout=0.2,activation="gelu"):
        super().__init__()

        
        self.model=[]
        for h in hidden:
            self.model.append(nn.LazyLinear(h))
            self.model.append(get_activation(activation))


        self.model.append(nn.Dropout(dropout))
        
        
        self.model.append(nn.LazyLinear(latent_dim))

        self.model=nn.Sequential(*self.model)
    
    def forward(self,x):
        return self.model(x)



def train_brain_epoch(model, train_dataloader, criterion=None, optim=None, device="cpu"):
    """
    Trains the model for one epoch using the specified dataloader and optimizer.
    
    Parameters:
    - model: PyTorch model to be trained
    - train_dataloader: Dataloader for the training data
    - criterion: Loss function to be used for training
    - optim: Optimizer to be used for training
    - device: Device to run the model and data on (CPU or GPU)
    
    Returns:
    - mean_loss: Mean loss of the model over the training data
    """
    model.train()
    
    loss_tmp = []
    # Iterate over the training data
    for x, y in train_dataloader:
        # Zero the gradients of the optimizer
        optim.zero_grad()
        # Move the data to the specified device
        x, y = x.to(device), y.to(device)
        # Get the model's prediction for the input data
        y_pred = model(x)
        
        # Calculate the loss based on the criterion and model's prediction
        if isinstance(criterion, nn.CosineEmbeddingLoss):
            # Set the target to 1 if using the CosineEmbeddingL
            target=torch.ones(y_pred.shape[0]).to(device)

            loss=criterion(y_pred.squeeze(),y.squeeze(),target)

        else:
            
            loss=criterion(y_pred.squeeze(),y.squeeze())
        # Backpropagate the loss and update the model's parameters
        loss.backward()
        optim.step()
        # Add the loss to the list of losses
        loss_tmp.append(loss.item())
    # Calculate the mean loss over all the data
    mean_loss = np.mean(loss_tmp)
    return mean_loss


def val_brain_epoch(model, val_dataloader, criterion=None, optim=None, device="cpu"):
    """
    Validates the model using the specified dataloader.
    
    Parameters:
    - model: PyTorch model to be validated
    - val_dataloader: Dataloader for the validation data
    - criterion: Loss function to be used for validation
    - optim: Optimizer to be used for validation (not used in this function)
    - device: Device to run the model and data on (CPU or GPU)
    
    Returns:
    - mean_loss: Mean loss of the model over the validation data
    """
    model.eval()
    
    loss_tmp = []
    i = 0
    # Iterate over the validation data
    with torch.no_grad():
        for x, y in val_dataloader:
            # Move the data to the specified device
            x, y = x.to(device), y.to(device)
            # Get the model's prediction for the input data
            y_pred = model(x)
            # Calculate the loss based on the criterion and model's prediction
            if isinstance(criterion, nn.CosineEmbeddingLoss):
                # Set the target to 1 if using the CosineEmbeddingLoss criterion
                target = torch.ones(y_pred.shape[0]).to(device)
                loss = criterion(y_pred.squeeze(), y.squeeze(), target)
            else:
                loss = criterion(y_pred.squeeze(), y.squeeze())
            # Add the loss to the list of losses
            loss_tmp.append(loss.item())
    # Calculate the mean loss over all the data
    mean_loss = np.mean(loss_tmp)
    return mean_loss
