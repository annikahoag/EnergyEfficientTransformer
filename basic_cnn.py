# NEXT STEPS -->
#   1. Create a GitHub repo and clean up code
#   2. Add in GPU tracking 
#   3. Add database collection properly so it's saving literally everything from the config file as well as param count and GPU and accuracy
#   4. Start actually finding good models 



#-------------------------Params-----------------------------
# 3 input channels, 32 output channels
# 3x3 kernel size
# Padding 1
# MaxPool2d
# 3 convolutional and pooling layers with feature map size 4x4
# No batch normalization 
# No dropout 
# ReLU activation function 
# Learning rate 1e-3
# Batch size 128 (training)
# Adam optimizer
# No weight decay
# 10 epochs

import torch 
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import yaml
import os
import sqlite3


# Read in YAML config file and set up global variables 
with open("Handcrafted_NNs/template.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# IN_CHANNELS = cfg["model"]["params"]["in_channels"]
# OUT_CHANNELS = cfg["model"]["params"]["out_channels"]
NUM_CLASSES = cfg["model"]["params"]["num_classes"]
CONV_CHANNELS = cfg["model"]["params"]["conv_channels"]
NUM_CONV = len(CONV_CHANNELS)-1
# NUM_FEATURES =  cfg["model"]["params"]["num_features"]
KERNEL_SIZE = cfg["model"]["params"]["kernel_size"]
IM_HEIGHT_WIDTH = cfg["image"]["im_height_width"]
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

PADDING = KERNEL_SIZE//2
FINAL_SPATIAL_DIM = IM_HEIGHT_WIDTH // (POOL_KERNEL ** NUM_CONV)
print()



class CNN(nn.Module):
    # Conv2d(in_channels, out_channels, kernel_size, padding), padding 1 keeps the spatial dimensions the same
    def __init__(self):
        super().__init__()

        # Convolutional and batch layers
        self.convs = nn.ModuleList()
        self.batches = nn.ModuleList()
        for i in range(NUM_CONV):
            self.convs.append(nn.Conv2d(CONV_CHANNELS[i], CONV_CHANNELS[i+1], KERNEL_SIZE, PADDING))
            if USE_BATCHNORM:
                self.batches.append(nn.BatchNorm2d(CONV_CHANNELS[i+1])) 

        # Instantiate activation function
        act_func = getattr(nn, ACTIVATION_FUNCTION)
        self.act = act_func(inplace=True)
        # # 1st convolutional layer, 3 input channels (RGB), 32 output channels, 3x3 kernel
        # self.conv1 = nn.Conv2d(IN_CHANNELS, CONV_CHANNELS[0], KERNEL_SIZE, PADDING)
        # if USE_BATCHNORM:
        #     self.bn1 = nn.BatchNorm2d(CONV_CHANNELS[0])

        # # 2nd convolutional layer, 32 to 64
        # self.conv2 = nn.Conv2d(CONV_CHANNELS[0], CONV_CHANNELS[1], KERNEL_SIZE, PADDING)
        # if USE_BATCHNORM:
        #     self.bn2 = nn.BatchNorm2d(CONV_CHANNELS[1])

        # # 3rd convolutional layer, 64 to 128
        # self.conv3 = nn.Conv2d(CONV_CHANNELS[1], CONV_CHANNELS[2], KERNEL_SIZE, PADDING)
        # if USE_BATCHNORM:
        #     self.bn3 = nn.BatchNorm2d(CONV_CHANNELS[2])

        # Pooling function
        PoolClass = getattr(nn, POOL_FUNC)
        if POOL_FUNC == 'MaxPool2d':
            self.pool = PoolClass(POOL_KERNEL)
        
        elif POOL_FUNC == 'AvgPool2d':
            self.pool = PoolClass(POOL_KERNEL)

        elif POOL_FUNC == 'LPPool2d':
            self.pool = PoolClass(NORM_TYPE_LPPOOL, POOL_KERNEL)
        
        elif POOL_FUNC == 'FractionalMaxPool2d':
            self.pool = PoolClass(OUTPUT_SIZE)
        
        elif POOL_FUNC == 'AdaptiveMaxPool':
            self.pool = PoolClass(OUTPUT_SIZE)
        
        elif POOL_FUNC == 'AdaptiveAvgPool':
            self.pool = PoolClass(OUTPUT_SIZE)

        # After 3 conv&pool layers, feature map size is 4×4 (starting from 32×32):
            # Input  → conv1, ReLU → pool (32→16)
            #        → conv2, ReLU → pool (16→8)
            #        → conv3, ReLU → pool (8→4)
        # So final tensor is (batch_size, 128, 4, 4) → flatten to 128*4*4 = 2048 features
        # TODO: fix this +
        self.fc = nn.Linear(CONV_CHANNELS[NUM_CONV]*FINAL_SPATIAL_DIM, NUM_CLASSES)


    # max_pool2d(x 2) halves height and width
    def forward(self, x):
        # print("convs: ", self.convs)
        # print("batches: ", self.batches)
        for i in range(NUM_CONV):
            # Convolutional layer
            # print(self.convs[i])
            x = self.convs[i](x)
            # print(x)
            if USE_BATCHNORM:
                x = self.batches[i](x)
            # x = nn.Conv2d(CONV_CHANNELS[i], CONV_CHANNELS[i+1], KERNEL_SIZE, PADDING)
            # if USE_BATCHNORM:
            #     x = nn.BatchNorm2d(CONV_CHANNELS[i+1])

            # Activation function 
            x = self.act(x)

            # Pooling
            x = self.pool(x)


        # # x: batch_size, 3, 32, 32
        # # x = nn_functional.ACTIVATION_FUNCTION(self.conv1(x)) #batch, 32, 32, 32
        # x = self.conv1(x)
        # x = self.act(x)
        # # x = nn_functional.max_pool2d(x, POOL_KERNEL) #batch, 32, 16, 16

        # # x = nn_functional.ACTIVATION_FUNCTION(self.conv2(x)) #batch, 64, 16, 16
        # x = self.conv1(x)
        # x = self.act(x)
        # # x = nn_functional.max_pool2d(x, POOL_KERNEL) #batch, 64, 8, 8

        # # x = nn_functional.ACTIVATION_FUNCTION(self.conv3(x)) #batch, 128, 8, 8
        # x = self.conv1(x)
        # x = self.act(x)
        # # x = nn_functional.max_pool2d(x, POOL_KERNEL) #batch, 128, 4, 4
        
        x = x.view(x.size(0), -1) #Flatten; batch, 128, 4, 4
        logits = self.fc(x) #batch, num_classes

        return logits



# Creates SQLite database and table
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


    # Defining transforms to be used for training and testing 
    # Data augmentation, converting to Tensor, normalization 
    transform_list = []
    if HORIZONTAL_FLIP:
        transform_list.append(transforms.RandomHorizontalFlip())
    if RANDOM_CROP:
        transform_list.append(transforms.RandomCrop(IM_HEIGHT_WIDTH, PADDING))
    transform_list += [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    transform_train = transforms.Compose(transform_list)

    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])


    # Downloading CIFAR-10 (50K train, 10K test; 10 classes, 32x32x3  images)
    DatasetClass = getattr(dsets, DATASET, None)
    # trainset = torchvision.datasets.DATASET(
    #     root = "./data", train=True, download=True, transform=transform_train
    # )
    trainset = DatasetClass(root=".\data", train=True, download=True, transform=transform_train)
    testset = DatasetClass(root=".\data", train=False, download=True, transform=transform_test)
    # testset = torchvision.datasets.DATASET(
    #     root="./data", train=False, download=True, transform=transform_test
    # )


    # Wrap in DataLoader, used for batching and shuffling
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE_TEST, shuffle=SHUFFLE_TEST, num_workers=NUM_WORKERS
    )


    # Get image class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




    # Loss, optimizer, and device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss() #For multi-class classification
    optimizer =getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # optimizer = optim.OPTIMIZER(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)



    # Typical training loop
    # num_epochs = 10

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(trainloader):

            # Move data to same device as the model 
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs) #batch, 10
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
                    f"Step [{batch_idx+1}/{len(trainloader)}], "
                    f"Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        

        # Evaluation after each epoch
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