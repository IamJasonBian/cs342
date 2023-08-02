from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

    
    train_data = load_detection_data('dense_data/train', num_workers=4)
    valid_data = load_detection_data('dense_data/valid', num_workers=4)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        for img, label,_ in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_train = torch.mean(loss(logit, label))
            
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_train, global_step)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        
        for img, label,_ in valid_data:
            img, label = img.to(device), label.to(device)
            logit = model(img)
            loss_valid = torch.mean(loss(logit, label))


        if valid_logger is not None:
            log(valid_logger, img, label, logit, global_step)


        if valid_logger is None or train_logger is None:
            print('epoch %-3d  \t train loss = %0.3f \t val loss = %0.3f' %
                  (epoch,loss_train,loss_valid))
        save_model(model)


    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
