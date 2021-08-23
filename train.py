

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import os
import PIL
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from torch import nn

PATH = './flower_mobilenet.pth'

#%%##########
# transform #
#############
img_size = 32
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
flowers_transform = T.Compose([T.Resize((img_size, img_size)),
                       T.RandomCrop(32, padding=4, padding_mode='reflect'),
                       T.RandomHorizontalFlip(),
                       T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                       T.ToTensor(),
                       T.Normalize(*stats,inplace=True)])


#%%#################
# make dataloaders #
####################
classes = ('daisy', 'dandelion', 'roses', 'sunflowers','tulips')

DATA_PATH_TRAINING_LIST = glob('./../data/flower_split/train/*/*.jpg')
DATA_PATH_TESTING_LIST = glob('./../data/flower_split/test/*/*.jpg')
batch_size = 16
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
        DATA_PATH_TESTING_LIST, 
        classes,
        transform=flowers_transform
    ),
    batch_size=batch_size,
    shuffle = True
)


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
loss_func = losses.crossEntropyLoss()



#%%###############################
# define optimizer and scheduler #
##################################
opt = optim.Adam(model.parameters(), lr=0.01)

lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

#%%#########
# training #
############
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

def show_accuracy(model, history, data_loader):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in data_loader:
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
        history[classname].append(accuracy)

def training(model, epoch):
    
    loss_history = {'train': []}
    test_accuracy_history = {'daisy': [], 'dandelion': [], 'roses': [], 'sunflowers': [],'tulips': []}
    valid_accuracy_history = {'daisy': [], 'dandelion': [], 'roses': [], 'sunflowers': [],'tulips': []}

    for epoch in range(epoch):
        loss_sum = 0
        loss_count = 0
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print(loss.item())
            loss_sum += loss.item()
            loss_count += 1
            # loss_history['train'].append(loss.item())

        loss_history['train'].append(loss_sum/loss_count)
        print("epoch: {:3d}, {:.2f} ".format(epoch,  loss_sum/loss_count))
        show_accuracy(model, test_accuracy_history, train_loader)
        show_accuracy(model, valid_accuracy_history, valid_loader)
        


        torch.save(model.state_dict(), PATH)
    return model, loss_history, test_accuracy_history, valid_accuracy_history

# %%
epoch=100
_, loss_hist, train_accu_hist, valid_accu_test = training(model, epoch)
# %%
# train-val progress
# num_epochs = params_train['num_epochs']
num_epochs = len(loss_hist['train'])

# plot loss progress
plt.title('Train Loss')
plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
# plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
plt.ylabel('loss avg')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.title('Train Accuracy')
plt.plot(range(1, num_epochs+1), train_accu_hist['daisy'], label='daisy')
plt.plot(range(1, num_epochs+1), train_accu_hist['dandelion'], label='dandelion')
plt.plot(range(1, num_epochs+1), train_accu_hist['roses'], label='roses')
plt.plot(range(1, num_epochs+1), train_accu_hist['sunflowers'], label='sunflowers')
plt.plot(range(1, num_epochs+1), train_accu_hist['tulips'], label='tulips')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.title('Valid Accuracy')
plt.plot(range(1, num_epochs+1), valid_accu_test['daisy'], label='daisy')
plt.plot(range(1, num_epochs+1), valid_accu_test['dandelion'], label='dandelion')
plt.plot(range(1, num_epochs+1), valid_accu_test['roses'], label='roses')
plt.plot(range(1, num_epochs+1), valid_accu_test['sunflowers'], label='sunflowers')
plt.plot(range(1, num_epochs+1), valid_accu_test['tulips'], label='tulips')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#%%############
# show result #
###############

dataiter = iter(valid_loader)
# print(dataiter)
images_and_labels = dataiter.next()
images, labels = images_and_labels[0].to(device), images_and_labels[1].to(device)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# show images
imshow(torchvision.utils.make_grid(images_and_labels[0]))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))
# print('[GroundTruth]')
# for i in range(4):
#     for j in range(8):
#         print(classes[labels[i*4+j]][0:6], end='\t')
#     print()




# %%
model = networks.mobilenet(alpha=1, num_classes=5).to(device)
model.load_state_dict(torch.load(PATH))
# %%
outputs = model(images)
# %%
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(8)))
# %%
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


with torch.no_grad():
    for data in valid_loader:
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

