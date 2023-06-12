from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb

def train(args):

    logger = tb.SummaryWriter('logs', flush_secs=1)

    #hardcoded params:
    dataset_path = 'data/train'

    #cast args to float

    '''
    args.learning_rate = float(args.learning_rate)
    args.num_epochs = int(args.num_epochs)
    args.batch_size = args.batch_size
    '''

    args.learning_rate = 0.05
    dataset_path = 'data/train'
    args.batch_size = 16
    args.num_epochs = 10    

    model = model_factory[args.model]()
    loss_fn = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # Load the data
    train_loader = load_data(dataset_path, args.batch_size)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Iterate over the training batches
        for images, labels in train_loader:

            #normalize gradient after use
            optimizer.zero_grad()

            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += accuracy(output, labels)

            logger.add_scalar('loss', loss.item(), epoch)
            logger.add_scalar('accuracy', accuracy(output, labels), epoch)

        # Calculate average loss and accuracy for the epoch
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader)

        print(f"Epoch {epoch + 1}/{args.num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Put custom arguments here (hyper parameters)
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    #parser.add_argument('-lr', '--learning_rate', default=0.05)
    #parser.add_argument('-e', '--num_epochs', default=10)
    #parser.add_argument('-b', '--batch_size', default=128)


    args = parser.parse_args()
    train(args)
