from os import path
import torch
import torch.utils.tensorboard as tb

def test_logging(train_logger, valid_logger):
    for epoch in range(10):
        torch.manual_seed(epoch)
        
        avg_train_accuracy = 0.
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)

            # Compute the global step
            global_step = epoch * 20 + iteration

            # Log the training loss
            train_logger.add_scalar('loss', dummy_train_loss, global_step)
            
            # Accumulate the training accuracy for average
            avg_train_accuracy += torch.mean(dummy_train_accuracy).item()

            # Log the training accuracy at the end of each epoch
            if iteration == 19:
                train_logger.add_scalar('accuracy', avg_train_accuracy / 20., global_step)

        torch.manual_seed(epoch)
        avg_validation_accuracy = 0.
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            
            # Compute the global step for validation (considering there are 20 steps in each epoch)
            global_step = epoch * 20 + iteration + 10

            # Accumulate the validation accuracy for average
            avg_validation_accuracy += torch.mean(dummy_validation_accuracy).item()

            # Log the validation accuracy at the end of each epoch
            if iteration == 9:
                valid_logger.add_scalar('accuracy', avg_validation_accuracy / 10., global_step)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)