from .models import CNNClassifier, save_model, model_factory
from os import path
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
import torch.utils.tensorboard as tb
import os

from .models import CNNClassifier, save_model, model_factory
from os import path
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
import torch.utils.tensorboard as tb
import os

def train(args):
    model = model_factory[args.model]()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    path = os.getcwd()
    print(path)

    train_logger = tb.SummaryWriter('logs/train')
    valid_logger = tb.SummaryWriter('logs/test')

    model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    train_data = load_data('dense_data/train')
    valid_data = load_data('dense_data/valid')

    # Initialize confusion matrix for train and validation
    cm_train = ConfusionMatrix(size=len(LABEL_NAMES))
    cm_valid = ConfusionMatrix(size=len(LABEL_NAMES))

    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = criterion(logit, label)
            acc_val = accuracy(logit, label)

            train_logger.add_scalar('loss', loss_val)
            train_logger.add_scalar('accuracy', acc_val)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())

            # Update the confusion matrix
            cm_train.add(logit.argmax(1), label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)

        model.eval()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            vacc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
            valid_logger.add_scalar('accuracy', acc_val)
            
            # Update the confusion matrix
            cm_valid.add(logit.argmax(1), label)

        avg_vacc = sum(vacc_vals) / len(vacc_vals)

        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))
        print('Confusion Matrix Training Class Accuracy:', cm_train.class_accuracy)
        print('Confusion Matrix Validation Class Accuracy:', cm_valid.class_accuracy)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    

    parser.add_argument('-m', '--model', choices=['cnn'], default='cnn')
    parser.add_argument('-n', '--num_epoch', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)