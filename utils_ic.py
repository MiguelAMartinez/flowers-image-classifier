import json
import torch
from torchvision import datasets, transforms
from PIL import Image

# Define function to read cat names
def read_jason(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

# Define function to read data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return trainloader, testloader, validloader, train_data

# Define processing testing image function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Resize and crop image
    im = Image.open(image)
    
    preprocess = transforms.Compose([transforms.Resize(255), 
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    im_tensor = preprocess(im)
    im_tensor.unsqueeze_(0)
    
    return im_tensor

# Define prediction function 
def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    img = img.to(device)
    
    output = model.forward(img)
    ps = torch.exp(output)    
    probs, idxs = ps.topk(topk)

    idx_to_class = dict((v,k) for k, v in model.classifier.class_to_idx.items())
    classes = [v for k, v in idx_to_class.items() if k in idxs.to('cpu').numpy()]
    
    if cat_to_name:
        classes = [cat_to_name[str(i + 1)] for c, i in \
                     model.classifier.class_to_idx.items() if c in classes]
        
    print('Probabilities:', probs.data.cpu().numpy()[0].tolist())
    print('Classes:', classes)