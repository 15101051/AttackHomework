from torchvision.models import resnet18

from models.BaseNormModel import BaseNormModel, Identity
from models.RobustBench.cifar10 import Andriushchenko2020Understanding, Carmon2019Unlabeled, Sehwag2020Hydra, Wang2020Improving, Hendrycks2019Using, Rice2020OverfittingNetL2
from attacks import VMI_FGSM
from tester import test_transfer_attack_acc, test_acc
from data.lesson import get_lesson_loader

test_acc(Identity(Andriushchenko2020Understanding(pretrained=True)), get_lesson_loader())
test_acc(BaseNormModel(resnet18(pretrained=True)), get_lesson_loader())  # acc = 0 because it's an imagenet model
attacker = VMI_FGSM([
    Identity(Sehwag2020Hydra(pretrained=True)), # model that requires normalization
    Identity(Andriushchenko2020Understanding(pretrained=True)) # model that do not need normalization
])

test_transfer_attack_acc(attacker,
                         get_lesson_loader(), 
                         [
                            # Identity(Andriushchenko2020Understanding(pretrained=True)), # white box attack
                            Identity(Wang2020Improving(pretrained=True)), # transfer attack
                         ],
                         save_path='./results/lesson/images/'
                         )