import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512]
num_epochs = 20
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
norm_layer = None #norm_layer = 'BN'
print(hidden_size)



#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
#################################################################################
# TODO: Q3.a Choose the right data augmentation transforms with the right       #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
import random
try:
    from PIL import ImageFilter
except ModuleNotFoundError:
    warn_missing_pkg('PIL', pypi_name='Pillow')
    _PIL_AVAILABLE = False
else:
    _PIL_AVAILABLE = True
class GaussianBlur(object):
    def __init__(self, sigma=(0.1, 2.0)):
        if not _PIL_AVAILABLE:
            raise ModuleNotFoundError( "error"
            )
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
# data_aug_transforms.extend([
# #             #transforms.RandomResizedCrop(50)
# #             #transforms.RandomResizedCrop(50, scale=(0.2, 1.))
#               #transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#               #transforms.RandomHorizontalFlip(p=0.5),
# #             transforms.RandomResizedCrop(100, scale=(0.5,0.9), ratio=(1, 1)),
#             # transforms.RandomApply([
#             #      transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             #  ], p=0.4),
#               #transforms.RandomGrayscale(p=0.1),
# #             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),

#          ])

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=test_transform
                                          )

#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#-------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
#-------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Define 5 convolutional blocks
        for i in range(len(hidden_layers)):

            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size[i - 1]

            layers += [nn.Conv2d(in_channels=input_dim, out_channels=hidden_size[i], kernel_size=3, stride=1, padding=1)]

            # We have commented the Dropout layer since we should run
            # either Dropout or BatchNorm at the same time

            # layers += [nn.Dropout(p=0.2)]
            layers += [nn.BatchNorm2d(hidden_size[i])]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers += [nn.ReLU()]

        # Collect convolutional blocks into sequential model
        self.Convlayers = nn.Sequential(*layers)
        # Define fully connected layer for classification
        self.Fconnected = nn.Linear(in_features=hidden_size[-1], out_features=num_classes)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Pass the batch to convolutional layers
        x = self.Convlayers(x)
        # Permute and pass the output of con. layer to fully connected layer
        x = x.permute(0, 2, 3, 1)
        x = self.Fconnected(x)
        # Flatten the output for computing loss purposes
        out = x.flatten(start_dim=1)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out



#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Filter and store parameter only if they are trainable
    trainable_parameters = filter(lambda is_trainable: is_trainable.requires_grad, model.parameters())
    # Compute the number of trainable parameters
    model_sz = sum([np.prod(parameter.size()) for parameter in trainable_parameters])
    print("Total number of parameters in the model is:", model_sz, sep=" ")

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz



#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image of stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract filters
    filters = list(model.parameters())[0]
    # N_filters
    n_filters = filters.size()[0]
    # Normalize
    filters = (filters - torch.min(filters)) / (torch.max(filters) - torch.min(filters))
    # Permute
    filters = filters.permute(0, 3, 2, 1)

    fig = plt.figure(figsize=(8, 8))
    rows = 8
    columns = 16
    # fig.tight_layout(pad=40)
    for i in range(n_filters):
        filter = filters[i, :, :, :]
        # Format
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.set_title('Filter n.{}'.format(i + 1), size=5)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # Plot
        plt.imshow(filter.cpu().detach().numpy())
    # Adjust margins
    plt.subplots_adjust(left=.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.6)
    plt.show()
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



#======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
#======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
#--------------------------------------------------------------------------------------
model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)
# Print the model
print(model)
# Print model size
#======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
#======================================================================================
PrintModelSize(model)
#======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
#======================================================================================
VisualizeFilter(model)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
loss_train = []
loss_val = []
best_accuracy = None
accuracy_val = []
best_model = type(model)(input_size, hidden_size, num_classes, norm_layer=norm_layer) # get a new instance
#best_model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer)
for epoch in range(num_epochs):

    model.train()

    loss_iter = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_iter += loss.item()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    loss_train.append(loss_iter/(len(train_loader)*batch_size))

    
    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    
        
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_iter = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            loss_iter += loss.item()
        
        loss_val.append(loss_iter/(len(val_loader)*batch_size))

        accuracy = 100 * correct / total
        accuracy_val.append(accuracy)
        print('Validation accuracy is: {} %'.format(accuracy))
        #################################################################################
        # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
        # the model with the best validation accuracy so-far (use best_model).          #
        #################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        patience = 7
        if best_accuracy == None:
            best_accuracy = accuracy
            counter = 0
        elif best_accuracy - accuracy < 0:
            best_accuracy = accuracy
            # reset counter
            counter = 0
        elif best_accuracy - accuracy > 0:
            counter += 1
            print(f"INFO: Early stopping counter {counter} of {patience}")
            if counter >= patience:
                print('INFO: Early stopping')
                torch.save(model, 'best_model.pt')
                torch.save(model.state_dict(), 'best_model.ckpt')
                print(f'Best model accuracy: {best_accuracy}')
                break

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()



plt.figure(2)
plt.plot(loss_train, 'r', label='Train loss')
plt.plot(loss_val, 'g', label='Val loss')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(accuracy_val, 'r', label='Val accuracy')
plt.legend()
plt.show()



#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# model = torch.load('best_model.pt')
# model.load_state_dict(torch.load('best_model.ckpt'))
# model.eval()


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#Compute accuracy on the test set
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))



# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
VisualizeFilter(model)



# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')


