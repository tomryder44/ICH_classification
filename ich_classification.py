
import numpy as np
import pandas as pd
from skimage import io
import os
import re
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import multilabel_confusion_matrix

# folder paths for training images
train_image_path = r''

# load the training labels into a pandas dataframe
labels = pd.read_csv('')

# quick summary of the dataset
print('-----------')
print('Dataset summary:')
print('Number of training images: %d' % len(labels))
print('Number of images with no ICH : %d' % (labels.loc[:,'epidural':] == 0).all(1).sum())
print('Number of images with 1 type: %d' % ((labels.loc[:,'epidural':] == 1).sum(1)==1).sum())
print('Number of images with 2 types: %d' % ((labels.loc[:,'epidural':] == 1).sum(1)==2).sum())
print('Number of images with 3 types: %d'% ((labels.loc[:,'epidural':] == 1).sum(1)==3).sum())
print('Number of epidural instances: %d' % labels['epidural'].sum())
print('Number of intraparenchymal instances: %d' % labels['intraparenchymal'].sum())
print('Number of subarachnoid instances: %d' % labels['subarachnoid'].sum())
print('-----------')

class TrainDataset(Dataset):
    '''Training dataset'''
    def __init__(self, root_dir, labels, transforms):
        self.root_dir = root_dir # images path
        self.labels = labels # dataframe of labels 
        self.transforms = transforms # image transformations
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        id_name = self.labels.loc[idx, 'ID'] # get id from index
        img_name = os.path.join(self.root_dir, id_name) + '.png' # image name from id 
        image = io.imread(img_name) # loaded as ndarray image 
        image = self.transforms(image) # apply transformations
        
        labels = self.labels.loc[idx, 'epidural':'subarachnoid'] # label for image
        labels = np.array(labels).astype('int')
        labels = torch.from_numpy(labels) # convert labels into tensor
            
        sample = {'image': image, 'labels': labels}
        return sample

def stats(dataset):
    '''Computes the mean and standard deviation of images in dataset.'''
    dataloader = DataLoader(dataset, batch_size=8,
                        shuffle=False, num_workers=0) 
    num_images = 0
    mean = 0.
    std = 0.
    for batch in dataloader:
        images = batch['image']
        images = images.view(images.size(0), images.size(1), -1)
        num_images += images.size(0)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= num_images
    std /= num_images
    print(mean.item())
    print(std.item())

# load dataset without normalisation to find mean and standard deviation of set
# transforms = transforms.Compose([transforms.ToTensor()])

# load dataset with normalisation
transforms = transforms.Compose([transforms.ToTensor(), 
                                 transforms.Normalize((0.185,), (0.303,))])

# create the dataset
dataset = TrainDataset(train_image_path, labels, transforms)
#stats(dataset) # find mean and std

def split(dataset, batch_size, split):
    '''Splits dataset into a training and validation set.'''
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size 
    # split dataset randomly into proportions train_size and val_size
    training_set, validation_set = torch.utils.data.random_split(dataset, 
                                                       [train_size, val_size])
    # check sizes of dataset are correct
    print('Size of dataset: %d' % len(dataset))
    print('Size of training set: %d' % train_size)
    print('Size of validation set: %d' % val_size)
    print('---------')
    return training_set, validation_set

batch_size = 16
trainset, validset = split(dataset, batch_size, split=0.9)

# Oversampling
def get_weights(dataset, labels):
    '''Get weight for each image for WeightedRandomSampler.'''
    # compute number of instances of each class
    num_instances = [labels['epidural'].sum(), 
                     labels['intraparenchymal'].sum(),
                     labels['subarachnoid'].sum()]
    
    # compute class weights (reciprocal of instances)
    weights = [1. / x for x in num_instances]
    
    # compute weight for each image 
    # assigned weight is that of class with least instances
    images_weights = []
    for img_label in np.array(labels):
        # get indices of positive cases for image
        img_idxs = [i for i, j in enumerate(img_label) if j==1]
        # get weight for image
        if len(img_idxs) > 0: # if image is not negative
            image_weight = max(weights[i] for i in img_idxs)
        else: # negative (no cases) images
            image_weight = 0 # still show some negative images
        images_weights.append(image_weight)
    return images_weights

# get weight for each image in training set
trainset_labels = labels.loc[trainset.indices, 'epidural':]    
images_weights = get_weights(trainset, trainset_labels) 

# get sampler
sampler = torch.utils.data.sampler.WeightedRandomSampler(
    images_weights, len(trainset))

# make loaders for both sets
trainloader = DataLoader(trainset, batch_size=batch_size,
                        shuffle=False, num_workers=0, sampler=sampler)  
validloader = DataLoader(validset, batch_size=batch_size,
                        shuffle=True, num_workers=0)  

# plot a batch of images
def plot_batch(loader):
    '''Plot grid of images in batch from a dataloader.'''
    # get a batch of images    
    batch = next(iter(loader)) 
    images = batch['image']
    # make a grid of images
    grid = torchvision.utils.make_grid(images)
    grid = grid.permute(1, 2, 0)
    grid = (0.303*grid) + 0.185 # unnormalize images
    plt.imshow(grid)
    plt.show()
 
# plot some images 
plot_batch(trainloader)
plot_batch(validloader)

class CNN(nn.Module):
    '''Convolutional neural network class.'''
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3) # greyscale image = 1 input channel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=64*14*14, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=3)
        self.activate = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''Forward pass of data through the network.'''
        x = self.pool(self.activate(self.conv1(x)))
        x = self.pool(self.activate(self.conv2(x)))
        x = self.pool(self.activate(self.conv3(x)))
        
        # flatten  x for fully connected layer
        #print(x.shape) # use to find fc1 in_features
        x = torch.flatten(x, start_dim=1)
        
        # pass through fully connected layer
        x = self.activate(self.fc1(x))
        x = self.fc2(x) 
        
        # pass output through a sigmoid function
        x = self.sigmoid(x)
        return x
        
def plot_training(train_loss, val_loss, train_recall, val_recall, train_precision, val_precision):
    '''Makes a subplot of loss, recall and precision vs epochs'''
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    xrange = range(1,len(train_loss)+1)
    
    ax1.plot(xrange, train_loss)
    ax1.plot(xrange, val_loss)
    ax1.set(ylabel='BCE')
    
    ax2.plot(xrange, train_recall)
    ax2.plot(xrange, val_recall)        
    ax2.set(ylabel='Recall')
    
    ax3.plot(xrange, train_precision, label='Training')
    ax3.plot(xrange, val_precision, label='Validation')        
    ax3.set(xlabel='Epochs', ylabel='Precision')
    
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.show()
    
def to_class(probabilities):
    return (probabilities>0.5).int()
    
def train(model, epochs=1000):
    '''Trains the model.'''
    
    # training metrics
    training_loss = [] # list of training loss for plotting
    training_recall = [] # list of training recall for plotting
    training_precision = [] # list of training precision for plotting
    
    # validation metrics
    validation_loss = [] # list of validation loss for plotting
    validation_recall = [] # list of validation recall for plotting
    validation_precision = [] # list of validation precision for plotting
        
    min_val_loss = 100 # for early stopping
    
    for epoch in range(1, epochs):
        
        model.train() # set model to train
        train_loss = 0. # running training loss
        train_recall = 0. 
        train_precision = 0. 
        label_count = torch.tensor([0,0,0]) # check class imbalance has been addressed 
        for i, train_data in enumerate(trainloader, 1): # training loop
            print('epoch: %d, batch: %d' % (epoch, i)) # print training progress
            images, labels = train_data['image'], train_data['labels'] # load data
            label_count += labels.sum(axis=0)
            optimizer.zero_grad() # zero the parameter gradients
            output = model(images) #  pass data through network
            class_output = to_class(output) # convert probs to classes
            loss = criterion(output, labels.float()) # compute loss
            loss.backward() # backpropagate the error
            optimizer.step() # take optimisation step
            # update metrics
            train_loss += (loss.item() * images.size(0)) # update running loss 
            train_recall += recall_score(labels, class_output, average='micro')
            train_precision += precision_score(labels, class_output, average='micro') 
        print('Running count of each class:', label_count.numpy())   
        
        model.eval() # set model for evaluation
        val_loss = 0. # running validation loss
        val_recall = 0.
        val_precision = 0.
        for valid_data in validloader: # validation loop
            images, labels = valid_data['image'], valid_data['labels']
            output = model(images)
            class_output = to_class(output)
            loss = criterion(output, labels.float()) # compute validation loss
            # update metrics
            val_loss += (loss.item() * images.size(0))
            val_recall += recall_score(labels, class_output, average='micro')
            val_precision += precision_score(labels, class_output, average='micro')
            
        # Compute training metrics for the epoch
        num_train_batches = len(trainloader)
        training_loss.append(train_loss/num_train_batches)
        training_recall.append(train_recall/num_train_batches)
        training_precision.append(train_precision/num_train_batches)
        
        # Compute validation metrics for the epoch
        num_val_batches = len(validloader)
        validation_loss.append(val_loss/num_val_batches)
        validation_recall.append(val_recall/num_val_batches)
        validation_precision.append(val_precision/num_val_batches)
        
        # Plot the metrics vs epochs
        plot_training(training_loss, validation_loss, 
                      training_recall, validation_recall,
                      training_precision, validation_precision)

        # implement early stopping
        patience = 10
        if (val_loss/num_val_batches) < min_val_loss:
            min_val_loss = (val_loss/num_val_batches)
            best_model = model
                                    
        if epoch > patience:
            if min_val_loss not in validation_loss[-patience:]:
                return best_model
            
# initialise network
cnn = CNN()

# training parameters
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=0.001)

# train the model
cnn = train(cnn)

def get_validation_metrics(model, loader):
    '''Test the model on validation data and compute recall, precision and confusion matrices.'''
    model.eval() # set model for evaluation
    recall = 0.
    precision = 0.
    confusion_matrices = np.zeros((3,2,2))
    for data in loader: # validation loop
        images, labels = data['image'], data['labels']
        output = model(images)
        class_output = to_class(output)
        recall += recall_score(labels, class_output, average='micro')
        precision += precision_score(labels, class_output, average='micro')
        confusion_matrices += multilabel_confusion_matrix(labels, class_output)
    num_batches = len(loader)
    recall = (recall/num_batches)*100
    precision = (precision/num_batches)*100
    print('Recall: %.2f%%' % recall)
    print('Precision: %.2f%%' % precision)
    print('Confusion matrices:', confusion_matrices)
        
# get final validation metrics for best model
get_validation_metrics(cnn, validloader)

# ~~~~~~~
# Testing 
# ~~~~~~~

test_image_path = r''
test_indices = range(4019)
filenames = [x for x in os.listdir(test_image_path)]
# found online to sort filenames numerically 
filenames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
test_idxs = [x[:-4] for x in filenames] # remove .png
test_submission_probs = pd.DataFrame()
test_submission_probs[''] = test_indices
test_submission_probs['ID'] = test_idxs
test_submission_classes = test_submission_probs.copy()

class TestDataset(Dataset):
    '''Coding challenge test set'''
    def __init__(self, root_dir, labels, transforms):
        self.root_dir = root_dir # images path
        self.labels = labels # dataframe of labels 
        self.transforms = transforms # image transformations
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        id_name = self.labels.loc[idx, 'ID'] # get id from index
        img_name = os.path.join(self.root_dir, id_name) + '.png' 
        image = io.imread(img_name) # ndarray image 
        image = self.transforms(image) # apply transformations
        sample = {'image': image}
        return sample

# Run on test set
testset = TestDataset(test_image_path, test_submission_probs, transforms)
testloader = DataLoader(testset, batch_size=1,
                        shuffle=False, num_workers=0)

def test_model(model, loader):
    '''Obtain test predictions.'''
    model.eval()
    outputs = []
    class_outputs = []
    for i, data in enumerate(loader):
        images = data['image']
        output = model(images)
        outputs.append(output)
        class_outputs.append(to_class(output))
    return outputs, class_outputs

probability_predictions, class_predictions = test_model(cnn, testloader)

# create probability predictions set
probability_predictions = torch.cat(probability_predictions, dim=0)
test_submission_probs['epidural'] = probability_predictions[:,0].detach().numpy()
test_submission_probs['intraparenchymal'] = probability_predictions[:,1].detach().numpy()
test_submission_probs['subarachnoid'] = probability_predictions[:,2].detach().numpy()

# create class predictions set
class_predictions = torch.cat(class_predictions, dim=0)
test_submission_classes['epidural'] = class_predictions[:,0].detach().numpy()
test_submission_classes['intraparenchymal'] = class_predictions[:,1].detach().numpy()
test_submission_classes['subarachnoid'] = class_predictions[:,2].detach().numpy()

# check
print(test_submission_probs.head())
print(test_submission_classes.head())

test_submission_probs.to_csv('test_predictions_probabilities.csv',index=False)
test_submission_classes.to_csv('test_predictions_classes.csv',index=False)