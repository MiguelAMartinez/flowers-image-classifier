import numpy as np
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

# Define classifier class
class NN_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

# Define validation function 
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Define NN function
def make_NN(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data):
    # Import pre-trained NN model 
    model = getattr(models, model_name)(pretrained=True)
    
    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False
        
    # Make classifier
    n_in = next(model.classifier.modules()).in_features
    n_out = len(labelsdict) 
    model.classifier = NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden)
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)

    model.to(device)
    start = time.time()

    epochs = n_epoch
    steps = 0 
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Training Loss: {:.3f} - ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} - ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
    
    # Add model info 
    model.classifier.n_in = n_in
    model.classifier.n_hidden = n_hidden
    model.classifier.n_out = n_out
    model.classifier.labelsdict = labelsdict
    model.classifier.lr = lr
    model.classifier.optimizer_state_dict = optimizer.state_dict
    model.classifier.model_name = model_name
    model.classifier.class_to_idx = train_data.class_to_idx
    
    print('model:', model_name, '- hidden layers:', n_hidden, '- epochs:', n_epoch, '- lr:', lr)
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    return model

# Define function to save checkpoint
def save_checkpoint(model, path):
    checkpoint = {'c_input': model.classifier.n_in,
                  'c_hidden': model.classifier.n_hidden,
                  'c_out': model.classifier.n_out,
                  'labelsdict': model.classifier.labelsdict,
                  'c_lr': model.classifier.lr,
                  'state_dict': model.state_dict(),
                  'c_state_dict': model.classifier.state_dict(),
                  'opti_state_dict': model.classifier.optimizer_state_dict,
                  'model_name': model.classifier.model_name,
                  'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)
    
# Define function to load model
def load_model(path):
    cp = torch.load(path)
    
    # Import pre-trained NN model 
    model = getattr(models, cp['model_name'])(pretrained=True)
    
    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False
    
    # Make classifier
    model.classifier = NN_Classifier(input_size=cp['c_input'], output_size=cp['c_out'], \
                                     hidden_layers=cp['c_hidden'])
    
    # Add model info 
    model.classifier.n_in = cp['c_input']
    model.classifier.n_hidden = cp['c_hidden']
    model.classifier.n_out = cp['c_out']
    model.classifier.labelsdict = cp['labelsdict']
    model.classifier.lr = cp['c_lr']
    model.classifier.optimizer_state_dict = cp['opti_state_dict']
    model.classifier.model_name = cp['model_name']
    model.classifier.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['state_dict'])
    
    return model

def test_model(model, testloader, device='cuda'):  
    model.to(device)
    model.eval()
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))