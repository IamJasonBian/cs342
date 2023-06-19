import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class LossType(str):
    """Standard names for loss type
    """
    SOFTMAX_CROSS_ENTROPY = "SoftmaxCrossEntropy"
    SOFTMAX_FOCAL_CROSS_ENTROPY = "SoftmaxFocalCrossEntropy"
    SIGMOID_FOCAL_CROSS_ENTROPY = "SigmoidFocalCrossEntropy"
    BCE_WITH_LOGITS = "BCEWithLogitsLoss"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX_CROSS_ENTROPY,
                         cls.SOFTMAX_FOCAL_CROSS_ENTROPY,
                         cls.SIGMOID_FOCAL_CROSS_ENTROPY,
                         cls.BCE_WITH_LOGITS])


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super(CNNClassifier, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        print("mid net")

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2704, 120)  # 5*5 from image dimension 
        
        print("late net")

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        
        print("forward")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ActivationType(str):
    """Standard names for activation type
    """
    SOFTMAX = "Softmax"
    SIGMOID = "Sigmoid"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX,
                         cls.SIGMOID])

    
class ClassificationLoss(torch.nn.Module):
    def __init__(self, class_weight=None,
                 loss_type=LossType.SOFTMAX_CROSS_ENTROPY):
        super(ClassificationLoss, self).__init__()

        self.loss_type = loss_type
        if loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
            self.criterion = torch.nn.CrossEntropyLoss(class_weight)
        elif loss_type == LossType.BCE_WITH_LOGITS:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise TypeError(
                "Unsupported loss type: %s. Supported loss type is: %s" % (
                    loss_type, LossType.str()))

    def forward(self, logits, target,
                use_hierar=False,
                is_multi=False,
                *argvs):
        device = logits.device
        if use_hierar:
            assert self.loss_type in [LossType.BCE_WITH_LOGITS,
                                      LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
            if not is_multi:
                target = torch.eye(self.label_size)[target].to(device)
            hierar_penalty, hierar_paras, hierar_relations = argvs[0:3]
            return self.criterion(logits, target) + \
                   hierar_penalty * self.cal_recursive_regularize(hierar_paras,
                                                                  hierar_relations,
                                                                  device)
        else:
            if is_multi:
                assert self.loss_type in [LossType.BCE_WITH_LOGITS,
                                          LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
            else:
                if self.loss_type not in [LossType.SOFTMAX_CROSS_ENTROPY,
                                          LossType.SOFTMAX_FOCAL_CROSS_ENTROPY]:
                    target = torch.eye(self.label_size)[target].to(device)
            return self.criterion(logits, target)


model_factory = {
    'cnn': CNNClassifier,
}

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
