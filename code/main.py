import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
# from torchsummary import summary

import torch
import torch.backends.cudnn # Backend for using NVIDIA CUDA
import numpy as np
import pickle

from torch import nn, optim
from torch.nn import functional as F
from torch.optim .optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils import data
from random import randint

import argparse
from pathlib import Path

# Enable benchmark mode on CUDNN since the input sizes do not vary. This finds the best algorithm to implement the convolutions given the layout.
torch.backends.cudnn.benchmark = True

# Add argument parser
parser = argparse.ArgumentParser(
    description="Training a 4-conv-layer CNN on UrbanSound8K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Add arguments to parse
parser.add_argument(
    "--log-dir",
    default=Path("logs"),
    type=Path
    )
parser.add_argument(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="Learning rate"
    )
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=5,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=300,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
)
parser.add_argument(
    "--dropout",
    default=0.5,
    type=float,
)
parser.add_argument(
    "--mode",
    default="LMC",
    type=str,
    help="The type of data to train the network on (LMC, MC, MLMC)"
)
parser.add_argument(
    "--optimiser",
    default="SGD",
    type=str,
    help="The optimiser used (SGD, Adam, AdamW)"
)
parser.add_argument(
    "--weight-decay",
    default=0.01,
    type=float,
    help="The L2 regularisation decay parameter"
)
parser.add_argument(
    "--TSCNN",
    action = 'store_true',
    help="Parameter for dealing with TSCNN combining of logits"
)
parser.add_argument(
    "--improvements",
    action = 'store_true',
    help="Parameter for adding improvements to the architecture CNN"
)

class DataShape(NamedTuple):
    height: int
    width: int
    channels: int

# Use GPU if cuda is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print ("Using CUDA...")
else:
    DEVICE = torch.device("cpu")
    print ("Using CPU...")


# Main function loop for training and testing the data
def main(args):

    # Load and prepare the data
    istrain = True
    train_dataset = UrbanSound8KDataset("./UrbanSound8K_train.pkl", istrain, args.mode, args.improvements)
    test_dataset = UrbanSound8KDataset("./UrbanSound8K_test.pkl", not istrain, args.mode, args.improvements)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    # Get the dimensions of the data
    data_channels = train_dataset.__getitem__(0)[0].shape[0]
    data_height = train_dataset.__getitem__(0)[0].shape[1]
    data_width = train_dataset.__getitem__(0)[0].shape[2]

    # Define the CNN model
    model = CNN(height=data_height, width=data_width, channels=data_channels, class_count=10, dropout=args.dropout, mode=args.mode, improvements=args.improvements)

    # Running Torch Summary to check the architecture
    # summary(model, (data_channels,data_height,data_width))

    # Define the unbalanced class weight of the data and move it to the appropriate device (hardcoded from analysis of the dataset)
    data_weight = torch.Tensor(6299/(np.array([6295,1825,6248,5121,5682,6282,1112,5886,5819,6299]))).to(DEVICE)

    # Define the criterion to be softmax cross entropy
    criterion = nn.CrossEntropyLoss(weight=data_weight)

    # Define the optimizer based on parsed arguments
    if args.optimiser == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, betas = (args.momentum, 0.999), weight_decay=args.weight_decay)
    elif args.optimiser == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate, betas = (args.momentum, 0.999), weight_decay=args.weight_decay)
    elif args.optimiser == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum, weight_decay=args.weight_decay)
    else:
        print("Error: Invalid optimiser argument, defaulting to SGD...")
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum, weight_decay=args.weight_decay)

    # Setup directory for the logs
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")

    # Define the summary writer for logging
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    # Prep notes file for reference
    f = open("logs/notes-sbatch.md", "a")
    f.write("Logged to: " + log_dir + (" - TSCNN" if args.TSCNN else f" - storing {args.mode}") + "\n")
    f.close()

    # Define the model trainer
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE, log_dir, args.TSCNN, args.mode
    )

    # Use the trainer to train the model
    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    # Close the summary writer at the end of the training
    summary_writer.close()

# The Dataset class
class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, istrain, mode, improvements):

        # Load the dataset
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode
        self.improvements = improvements
        self.istrain = istrain

    def __getitem__(self, index):

        # Extract the necessary features from the loaded dataset
        LM = self.dataset[index]["features"]["logmelspec"]
        MFCC = self.dataset[index]["features"]["mfcc"]
        C = self.dataset[index]["features"]["chroma"]
        SC = self.dataset[index]["features"]["spectral_contrast"]
        T = self.dataset[index]["features"]["tonnetz"]

        # Appropriately prepare the data given the selected mode, based on the specifications of the paper
        if self.mode == 'LMC':
            LMC = np.concatenate((LM, C, SC, T), axis=0)
            if self.improvements and self.istrain:
                LMC = LMC*randint(1,4) # Random augmentation of the training data
            feature = torch.from_numpy(LMC.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            MC = np.concatenate((MFCC, C, SC, T), axis=0)
            if self.improvements and self.istrain:
                MC = MC*randint(1,4) # Random augmentation of the training data
            feature = torch.from_numpy(MC.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            MLMC = np.concatenate((MFCC, LM, C, SC, T), axis=0)
            if self.improvements and self.istrain:
                MLMC = MLMC*randint(1,4) # Random augmentation of the training data
            feature = torch.from_numpy(MLMC.astype(np.float32)).unsqueeze(0)
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']

        return feature, label, fname, index

    def __len__(self):
        return len(self.dataset)

# The architecture class
class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float, mode: str, improvements: bool):
        super().__init__()

        # Define some global class variables
        self.input_shape = DataShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.mode = mode
        self.improvements = improvements

        # Defining the first convolutional layer & initialising its weights using Kaiming
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            bias=False,
            kernel_size=(3,3),
            padding=(1,1),
            # padding=(43,21),
            # stride=(2,2)
        )
        self.initialise_layer(self.conv1)

        # Defining batch normalisation of the outputs of the first conv layer
        self.bnorm1 = nn.BatchNorm2d(
            num_features=32
        )

        # Defining the second convolutional layer & initialising its weights using Kaiming
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (3, 3),
            bias=False,
            padding=(1,1),
            # padding = (43, 21),
            # stride=(2,2)
        )
        self.initialise_layer(self.conv2)

        # Defining batch normalisation of the outputs of the second conv layer
        self.bnorm2 = nn.BatchNorm2d(
            num_features = 32
        )

        # Defining the pooling layer for the batch normalised 2nd conv output
        self.pool2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            padding=(1,1),
            stride=(2, 2)
        )

        # Defining the third convolutional layer & initialising its weights using Kaiming
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3,3),
            bias=False,
            padding=(1,1),
            # padding = (22,11),
            # stride=(2,2)
        )
        self.initialise_layer(self.conv3)

        # Defining batch normalisation of the outputs of the third conv layer
        self.bnorm3 = nn.BatchNorm2d(
            num_features=64
        )

        # Defining the fourth convolutional layer & initialising its weights using Kaiming
        # Could use Max Pooling for the last layer, but probably more likely to be stride (based on the paper)
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            padding=(1,1),
            stride=(2,2),
            bias=False,
        )
        self.initialise_layer(self.conv4)

        # Adding a pooling layer with stride instead of conv4 with stride
        # self.pool4 = nn.MaxPool2d(
        #     kernel_size=(2, 2),
        #     padding=(1,1),
        #     stride=(2, 2)
        # )

        # Defining batch normalisation of the outputs of the fourth conv layer
        self.bnorm4 = nn.BatchNorm2d(
            num_features=64
        )

        # Defining the first fully connected layer & initialising the weights using Kaiming
        # The size of the data for MLMC is larger and requires a larger fully connected layer
        if self.mode == "MLMC":
            self.fc1 = nn.Linear(26048, 1024)
        else:
            self.fc1 = nn.Linear(15488, 1024)
        self.initialise_layer(self.fc1)

        # Defining batch normalisation of the outputs of the first fully connected layer
        self.bnormfc1 = nn.BatchNorm1d(
            num_features = 1024
        )

        # Defining the final fully connected layer to 10 classes & initialising the weights using Kaiming
        self.fc2 = nn.Linear (1024, 10)
        self.initialise_layer(self.fc2)

        # Defining the dropout used in the CNN
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:

        # Implementing the first conv hidden layer
        x = self.conv1(input_data)
        x = self.bnorm1(x)
        x = F.relu(x)

        # Implementing the second conv hidden layer
        x = self.conv2(self.dropout(x))
        x = self.bnorm2(x)
        x = F.relu(x)

        # Implementing a pooling stage to the outputs of the first layer
        x = self.pool2(x)

        # Implementing the third conv hidden layer
        # if self.improvements:
        #     x = self.dropout(x)
        x = self.conv3(x)
        x = self.bnorm3(x)
        x = F.relu(x)

        # Implementing the fourth conv hidden layer
        x = self.conv4(self.dropout(x))
        x = self.bnorm4(x)
        x = F.relu(x)

        # Use pooling with stride instead of conv4 with stride
        # x = self.pool4(x)

        # Flattening the output of the fourth conv layer for the first fc layer
        x = torch.flatten(x, start_dim = 1)

        # Implementing the first fully connected hidden layer
        x = self.fc1(self.dropout(x))
        # x = self.bnormfc1(x) # This was not in the paper
        x = torch.sigmoid(x)

        # Implementing the final fully connected hidden layer
        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

# Class for the execution of the main training loop
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        log_dir: str,
        TSCNN: bool,
        mode: str,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.log_dir = log_dir
        self.TSCNN = TSCNN
        self.mode = mode

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
    ):
        # Setting model to training mode
        self.model.train()

        # Defining list of results for each epoch
        results_epoch = {}

        # Main training loop
        for epoch in range(start_epoch, epochs):
            self.model.train()

            # Extracting required data from loader
            data_load_start_time = time.time()
            for batch, labels, fname, index in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                # Compute the forward pass of the model
                logits = self.model.forward(batch)

                # Calculate the loss of the forward pass
                loss = self.criterion(logits, labels)

                # Implement backpropogation
                loss.backward()

                # Update the optimiser parameters and set the update grads to zero again
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Disabling autograd when calculationg the accuracy
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                # Writing to logs and printing out the progress
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                # Update loop params for next batch
                self.step += 1
                data_load_start_time = time.time()

            # Write to summary writer at the end of each epoch
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                results_epoch[epoch] = self.validate(epoch, epochs, self.log_dir)

        # Exporting data
        if not self.TSCNN:
            pickle.dump(results_epoch, open("TSCNN_store_" + self.mode + ".pkl", "wb"))
        else:
            pickle.dump(results_epoch, open("TSCNN_store_" + "TSCNN" + ".pkl", "wb"))

    # Function used to print the progress
    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f},"

        )

    # Function used to log the progress
    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    # Function used to validate the model
    def validate(self, epoch, epochs, log_dir):
        results = {"preds": [], "labels": [], "logits": [], "indices": []}
        total_loss = 0

        # Loading data from previous runs and defining softmax to combine for TSCNN
        if self.TSCNN:
            results_epoch_LMC = pickle.load(open("TSCNN_store_LMC.pkl", "rb"))
            results_epoch_MC = pickle.load(open("TSCNN_store_MC.pkl", "rb"))
            smax = nn.Softmax(dim=-1)
            counter = 0  # used to load the appropriate logits from the stored data

        # Put model in validation mode
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels, fname, index in self.val_loader:

                # Shifting batch and labels to appropriate device for efficiency
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                # Calculating the logits of the testing batch
                logits = self.model(batch)

                # Averaging the logits by fname and making the new labels
                fname_logits, fname_labels, fname_indices = orderbyfname(labels,fname,logits, index)
                fname_logits = fname_logits.to(self.device)
                fname_labels = fname_labels.to(self.device)

                # Combining saved LMC and current MC logits for TSCNN, including some sanity checks
                if self.TSCNN:
                    combined_logits = []
                    if len(fname_logits) != len(fname_labels):
                        print("ERROR: Incorrect lengths of logit and label arrays, sanity check failed!") # Sanity check
                    for fname_logit, fname_label in zip(fname_logits, fname_labels):
                        if fname_label != results_epoch_LMC[epoch]["labels"][counter] or fname_label != results_epoch_MC[epoch]["labels"][counter]:
                            print("ERROR: Incorrect label, sanity check failed!") # Sanity check
                        combined_logits.append(np.array(smax(results_epoch_LMC[epoch]["logits"][counter]).cpu() + smax(results_epoch_MC[epoch]["logits"][counter]).cpu()))
                        counter += 1

                    # Overwriting logits with new combined logits and preparing the tensor
                    fname_logits = (torch.Tensor(combined_logits).type(torch.float)).to(self.device)


                # Calculating loss with new logits and labels
                loss = self.criterion(fname_logits, fname_labels)
                total_loss += loss.item()

                # Getting predictions of new logits
                preds = fname_logits.argmax(dim=-1).cpu().numpy()

                # Appending results
                results["logits"].extend(list(fname_logits))
                results["preds"].extend(list(preds))
                results["labels"].extend(list(fname_labels.cpu().numpy()))
                results["indices"].extend(list(fname_indices))

        # Find the overall accuracy of the model
        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )

        # Find the class accuracy of the model
        class_accuracy = compute_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )

        # Find the average class accuracy
        class_accuracy_avg = sum(class_accuracy)/100*len(class_accuracy)

        # Compute the average loss
        average_loss = total_loss / len(self.val_loader)

        # Write the progress to the logs
        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "average_class",
                {"test": class_accuracy_avg},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )

        # Switch model back to evaluation mode
        self.model.train()

        # Print the progress & exporting the softmaxed logits and labels
        display_text = f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}, class_accuracy: {class_accuracy}\nclass_avg: {class_accuracy_avg}"
        if (epoch+1) == epochs:
            f = open("logs/accuracy.md", "a")
            f.write(log_dir + "\n")
            f.write(display_text + "\n\n")
            f.close()
        print(display_text)
        return results

# Function for averaging the logits and proucing new labels from old
def orderbyfname(labels,fname,logits, index):
    fname_set = sorted(set(fname))
    new_logits = []
    new_labels = []
    new_indices = []
    for iter,name in enumerate(fname_set):

        # Determining the indices of the batch which are from the same filename
        indices = np.where(np.array(fname)==name)[0]
        sum = np.zeros(10)
        index_store_temp = []

        # Using the indices to average the logits
        for i in indices:
            sum += np.array(logits[i].cpu())
            index_store_temp.append(index[i]) # qualitative analysis
        sum = sum/len(indices)

        # appending new data
        new_logits.append(sum)
        new_labels.append(labels[indices[0]])

        # Storing the actual test data indices for the qualitative analysis
        new_indices.append(index_store_temp)

    return torch.Tensor(new_logits).type(torch.float), torch.Tensor(new_labels).type(torch.long), new_indices

# Function for computing the overall accuracy of the model
def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

# Function for computing the class accuracy of the model
def compute_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray], class_count: int = 10) -> float:
    assert len(labels) == len(preds)
    class_accuracy = []
    for class_label in range(0,class_count):
        class_labels = np.where(labels == class_label, class_label, class_label)
        class_accuracy.append(float(np.logical_and((preds == class_labels),(labels == class_labels)).sum())*100 / np.array(labels == class_labels).sum())
    return class_accuracy



# Function for handling the directory for writing logs to
def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (f'CNN_bn_epochs={args.epochs}_dropout={args.dropout}_bs={args.batch_size}_optim={args.optimiser}_decay={args.weight_decay}_lr={args.learning_rate}_momentum={args.momentum}_mode=' + ("TSCNN" if args.TSCNN else args.mode) + ("_improvements_" if args.improvements else "") +'_run_')
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

# Running the Progamme
if __name__ == "__main__":
    start = time.time()
    main(parser.parse_args())
    print ("Total time taken: {}".format(time.time() - start))
