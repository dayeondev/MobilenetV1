

#%%#######
# import #
##########

import torchvision.transforms as T
import torchvision
import torch
from core import datasets, networks, losses
from glob import glob
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary
import copy
import os
import PIL
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from torch import nn
import torchvision.transforms.functional as F
from tools.general import plot_utils, pil_utils



PATH = './flower_mobilenet.pth'
random_seed = 1
torch.manual_seed(random_seed)

#%%##########
# transform #
#############
img_size = 64
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
flowers_transform = T.Compose([T.Resize((img_size, img_size)),
                       T.RandomCrop(img_size, padding=4, padding_mode='reflect'),
                       T.RandomHorizontalFlip(),
                       T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                       T.ToTensor(),
                       T.Normalize(*stats,inplace=True)])


#%%#################
# make dataloaders #
####################
classes = ('daisy', 'dandelion', 'roses', 'sunflowers','tulips')

DATA_PATH_TRAINING_LIST = glob('./../data/flower_split/train/*/*.jpg')
DATA_PATH_VALID_LIST = glob('./../data/flower_split/val/*/*.jpg')
batch_size = 32
train_loader = DataLoader(
    datasets.FlowersDataset(
        DATA_PATH_TRAINING_LIST, 
        classes,
        transform=flowers_transform
    ),
    batch_size=batch_size,
    shuffle = True
)
valid_loader = DataLoader(
    datasets.FlowersDataset(
        DATA_PATH_VALID_LIST, 
        classes,
        transform=flowers_transform
    ),
    batch_size=batch_size,
    shuffle = False
)
# track=0
# for data, label in train_loader:
#     track+=+1
#     print(track)
#     print(label.shape)
#     print(data.shape)
# print('stop')


#%%##############
# make networks #
#################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn((3, 3, 224, 224)).to(device)
model = networks.mobilenet(alpha=1, num_classes=5).to(device)
output = model(x)
print('output size:', output.size())

summary(model, (3, 224, 224), device=device.type)

#%%############
# define loss #
###############
criterion = losses.crossEntropyLoss()
loss_fn = nn.CrossEntropyLoss(reduction='sum')

#%%###############################
# define optimizer and scheduler #
##################################
learning_rate = 0.0001
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

epoch = 100


#%%#########
# training #
############

def get_accuracy(output, label):
    predict = output.argmax(1, keepdim=True)
    corrects = predict.eq(label.view_as(predict)).sum().item()
    return corrects

def calculate_batch(loss_fn, output, label, optimizer=None):
    # print(output)
    loss = loss_fn(output, label)
    accuracy = get_accuracy(output, label)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), accuracy

def calculate_loss_and_accuracy(model, loss_fn, data_loader, optimizer=None):
    sum_losses = 0.0
    sum_accuracy = 0.0
    len_dataset = len(data_loader.dataset)

    for input, label in data_loader:
        input = input.to(device)
        label = label.to(device)
        output = model(input)

        loss, accuracy = calculate_batch(loss_fn, output, label, optimizer)

        sum_losses += loss

        if accuracy is not None:
            sum_accuracy += accuracy
    
    
    return sum_losses/len_dataset, sum_accuracy/len_dataset


def train_valid(model, epochs):

    loss_history = {'train': [], 'valid': []}
    accuracy_history = {'train': [], 'valid': []}
    
    best_accuracy = float('-inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(epochs):
        print("epoch: {}, time: {}".format(epoch, (time.time() - start_time)/60))
        
        model.train()
        train_loss, train_accuracy = calculate_loss_and_accuracy(model, loss_fn, train_loader, optimizer)
        loss_history['train'].append(train_loss)
        accuracy_history['train'].append(train_accuracy)

        model.eval()
        with torch.no_grad():
            valid_loss, valid_accuracy = calculate_loss_and_accuracy(model, loss_fn, valid_loader)
        loss_history['valid'].append(valid_loss)
        accuracy_history['valid'].append(valid_accuracy)

                
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            # best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), PATH)
            print('Copied best model weights!')


        if epoch % 10 == 9:
            plot_utils.show_plot(loss_history, len(loss_history['train']), True)
            plot_utils.show_plot(accuracy_history, len(loss_history['train']), False)
            # plot_utils.show_accuracy(accuracy_history['train'], len(accuracy_history['train']['daisy']), True)
            # plot_utils.show_accuracy(accuracy_history['valid'], len(accuracy_history['valid']['daisy']), False)
        
    model.load_state_dict(best_model_weights)
    return loss_history, accuracy_history


loss_history , accuracy_history = train_valid(model, epoch)
# _, loss_hist = training(model, epoch)

#%%############
# show result #
###############
# %%
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


with torch.no_grad():
    for data in train_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # Check correct prediction
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))
# %%
with torch.no_grad():
    sum_accuracy = 0
    for data in valid_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        accuracy = get_accuracy(outputs, labels)
        if accuracy is not None:
            sum_accuracy += accuracy

    print(sum_accuracy / len(valid_loader.dataset))

# %%