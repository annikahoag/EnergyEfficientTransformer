# NEXT STEPS -->
#   1. Create a GitHub repo and clean up code (DONE)
#   2. Add in GPU tracking (DONE)
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
import subprocess 
import csv



# Read in YAML config file and set up global variables 
CONFIG_FILE = "template.yaml"
CONFIG_PATH = str("Handcrafted_NNs/" + CONFIG_FILE)
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg["model"]["name"]
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


# Output file for GPU readings and command to run during subprocess
OUTPUT_FILE_PER_EPOCH = 'power_output_per_epoch.csv' #So we can see how much each epoch used
OUTPUT_FILE_TOTAL = 'total_power.csv' #So we can see how much the entire training used 
COMMAND = ["nvidia-smi", "dmon", "-s", "p", "--format", "csv"]
DB_NAME = str(MODEL_NAME + '_results')
ALL_DB_NAME = 'final_results'

print()



# Class for Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()


        # Set up modules lists for convolutional layers and batch norm for in between each if needed 
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
def create_db(name):
    conn = sqlite3.connect(name)
    c = conn.cursor()

    # The DB that contains one row per model with all parameters listed and the wattage used for the entire training
    if name==ALL_DB_NAME:
        c.execute("""
            CREATE TABLE IF NOT EXISTS name (
                id INTEGER PRIMARY KEY,
                model_name STRING, 
                num_classes INTEGER, 
                conv_channels TEXT, 
                kernel_size INTEGER,
                pool_kernel INTEGER, 
                use_batchnorm INTEGER,
                activation_function STRING,
                pooling STRING,
                norm_type INTEGER, 
                output_size INTEGER,
                dataset STRING,
                random_crop INTEGER,
                horizontal_flip INTEGER, 
                batch_size_train INTEGER,
                num_epochs INTEGER,
                learning_rate REAL,
                weight_decay REAL,
                num_workers INT,
                shuffle_train INTEGER,
                optimizer STRING,
                batch_size_test INTEGER,
                shuffle_test INTEGER,
                im_height_width INTEGER,
                num_conv_layer INTEGER,
                padding INTEGER,
                final_spatial_dim INTEGER,
                num_params INTEGER,
                accuracy REAL,
                total_wattage REAL, 
                avg_wattage REAL,
                config_file BLOB
            )
        """)
    
    # The other DB just keeps track of each epochs' accuracy, etc. 
    else:
         c.execute("""
            CREATE TABLE IF NOT EXISTS name (
                epoch INTEGER PRIMARY KEY,
                model_name STRING, 
                num_params INTEGER,
                accuracy REAL,
                total_wattage REAL, 
                avg_wattage REAL,
                config_file BLOB
            )
        """)
    conn.commit()
    conn.close()



# Function to read CSV file with wattage readings, return total and average wattage used at this point 
def get_power(filename):
    with open(filename, 'r') as f:
        reader = list(csv.reader(f, delimiter=','))
        num_iters = len(reader)

    total = 0
    num_skips = 0

    for i in range(num_iters):
        if reader[i][0] != '#gpu' and reader[i][0] != '#Idx':
            total = total + float(reader[i][1])
        else:
            num_skips += 1
    
    avg = total / (num_iters-num_skips)
    
    return total, avg



# Append given data to our database 
# TODO: Same as create_db() 
def add_to_db(name, epoch, num_params, accuracy, total_wattage, avg_wattage):
    conn = sqlite3.connect(name)
    c = conn.cursor()

    if name==ALL_DB_NAME:
        c.execute('''
        INSERT OR REPLACE INTO name(
            model_name, 
            num_classes, 
            conv_channels, 
            kernel_size, 
            pool_kernel, 
            use_batchnorm, 
            activation_function, 
            pooling, 
            norm_type, 
            output_size, 
            dataset, 
            random_crop, 
            horizontal_flip,
            batch_size_train, 
            num_epochs,
            learning_rate,
            weight_decay,
            num_workers,
            shuffle_train, 
            optimizer, 
            batch_size_test,
            shuffle_test,
            im_height_width,
            num_conv_layer,
            padding,
            final_spatial_dim,
            num_params,
            accuracy,
            total_wattage,
            avg_wattage,
            config_file
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
            MODEL_NAME, 
            NUM_CLASSES, 
            str(CONV_CHANNELS), 
            KERNEL_SIZE, 
            POOL_KERNEL, 
            USE_BATCHNORM, 
            ACTIVATION_FUNCTION, 
            POOL_FUNC, 
            NORM_TYPE_LPPOOL, 
            OUTPUT_SIZE, 
            DATASET, 
            RANDOM_CROP, 
            HORIZONTAL_FLIP, 
            BATCH_SIZE_TRAIN,
            NUM_EPOCHS,
            LEARNING_RATE,
            WEIGHT_DECAY,
            NUM_WORKERS,
            SHUFFLE_TRAIN,
            OPTIMIZER,
            BATCH_SIZE_TEST,
            SHUFFLE_TEST,
            IM_HEIGHT_WIDTH, 
            NUM_CONV,
            PADDING,
            FINAL_SPATIAL_DIM,
            num_params,
            accuracy,
            total_wattage,
            avg_wattage,
            CONFIG_FILE
        ))

    else:
        c.execute('''
            INSERT OR REPLACE INTO name(
                epoch, 
                model_name,
                num_params,
                accuracy,
                total_wattage,
                avg_wattage,
                config_file
            )
            VALUES(?, ?, ?, ?, ?, ?, ?)
        ''', (
                epoch,
                MODEL_NAME, 
                num_params,
                accuracy,
                total_wattage,
                avg_wattage,
                CONFIG_FILE
            ))
    
    conn.commit()
    conn.close()




def main():
    # Only do this one time, creating SQLite database
    if not os.path.exists(DB_NAME):
        create_db(DB_NAME)
    if not os.path.exists(ALL_DB_NAME):
        create_db(ALL_DB_NAME)


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
    device = torch.device("cuda")
    print("Device ", device)
    model = CNN().to(device)

    # Loss and optimizer set up
    criterion = nn.CrossEntropyLoss() #For multi-class classification
    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    # Start process that will track the wattage for the entire training
    f1 = open(OUTPUT_FILE_TOTAL, "a")
    proc1 = subprocess.Popen(
        COMMAND,
        stdout = f1,
        stderr = subprocess.STDOUT,
        start_new_session = True, 
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    # Train model
    for epoch in range(NUM_EPOCHS):
        # Start GPU tracking subprocess for the specific epoch we're on 
        f2 = open(OUTPUT_FILE_PER_EPOCH, "a")
        proc2 = subprocess.Popen(
            COMMAND,
            stdout = f2,
            stderr = subprocess.STDOUT,
            start_new_session = True, 
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

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
        

        # Before testing save power, kill subprocess for this epoch, and close the output file so we can write to it again 
        total_power, avg_power = get_power(OUTPUT_FILE_PER_EPOCH)
        proc2.terminate()
        try:
            proc2.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc2.kill()
            proc2.wait()
        f2.close()
        os.remove(OUTPUT_FILE_PER_EPOCH)

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
        print(f"Total params: {param_count}")

        print(f"Total GPU usage: {total_power}\n")

        add_to_db(DB_NAME, epoch+1, param_count, acc, total_power, avg_power)


    # Save results for fully trained model
    total_power, avg_power = get_power(OUTPUT_FILE_TOTAL)
    proc1.terminate()
    try:
        proc1.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc1.kill()
        proc1.wait()
    f1.close()
    os.remove(OUTPUT_FILE_TOTAL)
    add_to_db(ALL_DB_NAME, NUM_EPOCHS, param_count, acc, total_power, avg_power)



if __name__ == "__main__":
    main()