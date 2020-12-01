'''
剪枝设置如下：
parameters_to_prune = (
    (model_ft.features[0], 'weight'),
    (model_ft.features[0], 'bias'),
    (model_ft.features[3], 'weight'),
    (model_ft.features[3], 'bias'),
    (model_ft.features[6], 'weight'),
    (model_ft.features[6], 'bias'),
    (model_ft.features[8], 'weight'),
    (model_ft.features[8], 'bias'),
    (model_ft.features[10], 'weight'),
    (model_ft.features[10], 'bias'),
    (model_ft.classifier[1], 'weight'),
    (model_ft.classifier[1], 'bias'),
    (model_ft.classifier[4], 'weight'),
    (model_ft.classifier[4], 'bias'),
    (model_ft.classifier[6], 'weight'),
    (model_ft.classifier[6], 'bias'),
)

1. 迭代30次
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.80,
)

conv1 sparsity: 17.88% (weight + bias)
conv2 sparsity: 35.08%
conv3 sparsity: 34.98%
conv4 sparsity: 39.33%
conv5 sparsity: 39.99%
fc1 sparsity: 84.88%
fc2 sparsity: 75.20%
fc3 sparsity: 85.65%

结果:0.799127

2. 迭代30次
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.85,
)
conv1 sparsity: 20.08%
conv2 sparsity: 39.24%
conv3 sparsity: 39.35%
conv4 sparsity: 44.15%
conv5 sparsity: 44.90%
fc1 sparsity: 89.67%
fc2 sparsity: 80.66%
fc3 sparsity: 97.49%
结果：
0.748770

3. 迭代50次(就不继续往上测了)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.85,
)
conv1 sparsity: 20.08%
conv2 sparsity: 39.24%
conv3 sparsity: 39.35%
conv4 sparsity: 44.15%
conv5 sparsity: 44.90%
fc1 sparsity: 89.67%
fc2 sparsity: 80.66%
fc3 sparsity: 97.49%
结果：0.791984


'''

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "../data/NWPU-RESISC45-20%"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "alexnet"

# Number of classes in the dataset
num_classes = 45

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
num_epochs = 30

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 256

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    model_ft = nn.DataParallel(model_ft)
# Send the model to GPU
model_ft = model_ft.to(device)

parameters_to_prune = (
    (model_ft.features[0], 'weight'),
    (model_ft.features[0], 'bias'),
    (model_ft.features[3], 'weight'),
    (model_ft.features[3], 'bias'),
    (model_ft.features[6], 'weight'),
    (model_ft.features[6], 'bias'),
    (model_ft.features[8], 'weight'),
    (model_ft.features[8], 'bias'),
    (model_ft.features[10], 'weight'),
    (model_ft.features[10], 'bias'),
    (model_ft.classifier[1], 'weight'),
    (model_ft.classifier[1], 'bias'),
    (model_ft.classifier[4], 'weight'),
    (model_ft.classifier[4], 'bias'),
    (model_ft.classifier[6], 'weight'),
    (model_ft.classifier[6], 'bias'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.80,
)

print(
    "conv1 sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model_ft.features[0].weight == 0)
            + torch.sum(model_ft.features[0].bias == 0)
        )
        / float(
            model_ft.features[0].weight.nelement()
            + model_ft.features[0].bias.nelement()
        )
    )
)

print(
    "conv2 sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model_ft.features[3].weight == 0)
            + torch.sum(model_ft.features[3].bias == 0)
        )
        / float(
            model_ft.features[3].weight.nelement()
            + model_ft.features[3].bias.nelement()
        )
    )
)


print(
    "conv3 sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model_ft.features[6].weight == 0)
            + torch.sum(model_ft.features[6].bias == 0)
        )
        / float(
            model_ft.features[6].weight.nelement() 
            + model_ft.features[6].bias.nelement()
        )
    )
)

print(
    "conv4 sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model_ft.features[8].weight == 0)
            + torch.sum(model_ft.features[8].bias == 0)
        )
        / float(
            model_ft.features[8].weight.nelement()
            + model_ft.features[8].bias.nelement()
        )

    )
)

print(
    "conv5 sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model_ft.features[10].weight == 0)
            + torch.sum(model_ft.features[10].bias == 0)
        )
        / float(
            model_ft.features[10].weight.nelement()
            + model_ft.features[10].bias.nelement()
        )
    )
)

print(
    "fc1 sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model_ft.classifier[1].weight == 0)
            + torch.sum(model_ft.classifier[1].bias == 0)
        )
        / float(
            model_ft.classifier[1].weight.nelement()
            + model_ft.classifier[1].bias.nelement()
        )
    )
)

print(
    "fc2 sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model_ft.classifier[4].weight == 0)
            +  torch.sum(model_ft.classifier[4].bias == 0)
        )
        / float(
            model_ft.classifier[4].weight.nelement()
            + model_ft.classifier[4].bias.nelement()
        )
    )
)

print(
    "fc3 sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model_ft.classifier[6].weight == 0)
            + torch.sum(model_ft.classifier[6].bias == 0)
        )
        / float(
            model_ft.classifier[6].weight.nelement()
            +model_ft.classifier[6].bias.nelement() 
        )
    )
)


print(dict(model_ft.named_buffers()).keys())
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))