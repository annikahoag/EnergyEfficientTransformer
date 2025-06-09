# NEXT STEPS -->
#   1. Create a GitHub repo and clean up code (DONE)
#   2. Add in GPU tracking 
#   3. Add database collection properly so it's saving literally everything from the config file as well as param count and GPU and accuracy
#   4. Start actually finding good models 



# Import all necessary libraries 
import torch 
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sqlite3


# Read in YAML config file and set up global variables 
with open("Handcrafted_NNs/template.yaml", "r") as f:
    cfg = yaml.safe_load(f)

NUM_CLASSES = cfg["model"]["params"]["num_classes"]
CONV_CHANNELS = cfg["model"]["params"]["conv_channels"] #Formatted as [original input channel, layer1 output channel/layer2 input channel, layer2 output channel/layer3 input channel....]
KERNEL_SIZE = cfg["model"]["params"]["kernel_size"]
POOL_KERNEL = cfg["model"]['params']['pool_kernel'] #max pooling kernel is POOL_KERNELxPOOL_KERNEL
USE_BATCHNORM = cfg["model"]["params"]["use_batchnorm"]
ACTIVATION_FUNCTION = cfg["model"]["params"]["activation_function"]
POOL_FUNC = cfg["model"]["params"]["pooling"]
NORM_TYPE_LPPOOL = cfg["model"]["params"]["norm_type"]
OUTPUT_SIZE = cfg["model"]["params"]["output_size"]

DATASET = cfg["training"]["dataset"]
RANDOM_CROP = cfg["training"]["data_augmentation"]["random_crop"]
HORIZONTAL_FLIP = cfg["training"]["data_augmentation"]["horizontal_flip"]
BATCH_SIZE_TRAIN = cfg["training"]["batch_size"]
NUM_EPOCHS = cfg["training"]["num_epochs"]
LEARNING_RATE = float(cfg["training"]["learning_rate"])
WEIGHT_DECAY = float(cfg["training"]["weight_decay"])
NUM_WORKERS = cfg["training"]["num_workers"]
SHUFFLE_TRAIN = cfg["training"]["shuffle"]
OPTIMIZER = cfg["training"]["optimizer"]

BATCH_SIZE_TEST = cfg["testing"]["batch_size"]
SHUFFLE_TEST = cfg["testing"]["shuffle"]

IM_HEIGHT_WIDTH = cfg["image"]["im_height_width"]

NUM_CONV = len(CONV_CHANNELS)-1
PADDING = KERNEL_SIZE//2
FINAL_SPATIAL_DIM = IM_HEIGHT_WIDTH // (POOL_KERNEL ** NUM_CONV)
print()



# Class for Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()


        # Set up modules lists for convolutional and batch layers for in between each if needed 
        self.convs = nn.ModuleList()
        self.batches = nn.ModuleList()
        for i in range(NUM_CONV):
            self.convs.append(nn.Conv2d(CONV_CHANNELS[i], CONV_CHANNELS[i+1], KERNEL_SIZE, PADDING))
            if USE_BATCHNORM:
                self.batches.append(nn.BatchNorm2d(CONV_CHANNELS[i+1])) 


        # Set up activation function
        act_func = getattr(nn, ACTIVATION_FUNCTION)
        self.act = act_func(inplace=True)


        # Set up pooling function 
        PoolClass = getattr(nn, POOL_FUNC)
        if POOL_FUNC == 'MaxPool2d' or POOL_FUNC == 'AvgPool2d':
            self.pool = PoolClass(POOL_KERNEL)
        elif POOL_FUNC == 'LPPool2d':
            self.pool = PoolClass(NORM_TYPE_LPPOOL, POOL_KERNEL)
        elif POOL_FUNC == 'FractionalMaxPool2d' or POOL_FUNC == 'AdaptiveMaxPool' or POOL_FUNC == 'AdaptiveAvgPool':
            self.pool = PoolClass(OUTPUT_SIZE)
        

        # Apply linear transformation 
        # TODO: include bias parameter as an option?     
        self.fc = nn.Linear(CONV_CHANNELS[NUM_CONV]*FINAL_SPATIAL_DIM, NUM_CLASSES)



    # Function for forward pass through our CNN
    def forward(self, x):
        
        # Call each convolutional layer from init, apply batch norm if needed, apply activation function and pooling
        for i in range(NUM_CONV):
            x = self.convs[i](x)
            if USE_BATCHNORM:
                x = self.batches[i](x)
            x = self.act(x)
            x = self.pool(x)

        # Flatten        
        x = x.view(x.size(0), -1) 

        return self.fc(x) 



# Creates SQLite database and table
# TODO: Have to make this actually be what we want to put into database, right now it's a placeholder ish
def create_db():
    conn = sqlite3.connect('model_results')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name STRING, 
            param_count INTEGER,
            gpu_history BLOB,
            accuracy REAL,
            validation REAL,
            linear_layer_count INTEGER,
            conv_count INTEGER,
            architecture BLOB,
            batch_size INTEGER,
            optimizer STRING,
            dataset STRING,
            config_file BLOB
        )
    """)
    conn.commit()
    conn.close()



# Append given data to our database 
# TODO: Same as create_db() 
def add_to_db(id, model_num, param_count, gpu_history, accuracy, validation, linear_layer_count, conv_count, architecture, batch_size, optimizer, datatset, config_file):
    conn = sqlite3.connect('results')
    c = conn.cursor()

    c.execute('''
        INSERT OR REPLACE INTO results(id, model_name, param_count, gpu_history, accuracy, validation, linear_layer_count, conv_count, architecture, batch_size, optimizer, dataset, config_file)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (id, model_num, param_count, gpu_history, accuracy, validation, linear_layer_count, conv_count, architecture, batch_size, optimizer, dataset, config_file))
    
    conn.commit()
    conn.close()



def main():
    # Only do this one time, creating SQLite database
    if not os.path.exists('results.db'):
        create_db()


    # Setting up data

    # Defining transforms to be used for training and testing 
    # Data augmentation, converting to Tensor, normalization 
    transform_list = []
    if HORIZONTAL_FLIP:
        transform_list.append(transforms.RandomHorizontalFlip())
    if RANDOM_CROP:
        transform_list.append(transforms.RandomCrop(IM_HEIGHT_WIDTH, PADDING))
    transform_list += [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]

    transform_train = transforms.Compose(transform_list)
    transform_test=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


    # Get CIFAR-10 (50K train, 10K test; 10 classes, 32x32x3  images)
    DatasetClass = getattr(dsets, DATASET, None)
    trainset = DatasetClass(root=".\data", train=True, download=True, transform=transform_train)
    testset = DatasetClass(root=".\data", train=False, download=True, transform=transform_test)


    # Wrap in DataLoader, used for batching and shuffling
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=SHUFFLE_TEST, num_workers=NUM_WORKERS)


    # Get image class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    # Device set-up
    # TODO: Have to add in GPU usage tracking 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    # Loss and optimizer set up
    criterion = nn.CrossEntropyLoss() #For multi-class classification
    optimizer =getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    # Train model
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(trainloader):

            # Move data to same device as the model 
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate loss
            running_loss += loss.item()
            if (batch_idx+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
                    f"Step [{batch_idx+1}/{len(trainloader)}], "
                    f"Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        

        # Evaluate performance after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Test Accuracy: {acc:.2f}%")

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Total params: {param_count}\n")



if __name__ == "__main__":
    main()