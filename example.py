"""
Method: AttackSuccessRate FinalScore(ASR*SSIM) Submit Score
VMI_FGSM:                         0.479  0.4221 34.546
VMI_FGSM step=20:                 0.5275 0.4648
MI_CommonWeakness:                0.4845 0.4132
SGD:                              0.059  0.4648
VMI_Inner_CommonWeakness:         0.4945 0.209
VMI_Inner_CommonWeakness step=20: 0.5225 0.2225
VMI_Outer_CommonWeakness:         0.4825 0.4207
MI_FGSM:                          0.4545 0.403
Adam_CommonWeakness:              0.0805
MI_CosineSimilarityEncourager:    0.4925 0.2941
MI_RandomWeight:                  0.2855
"""
from torchvision.models import resnet18

from models.BaseNormModel import BaseNormModel, Identity
from models.RobustBench.cifar10 import Andriushchenko2020Understanding, Carmon2019Unlabeled, Sehwag2020Hydra, Wang2020Improving, Hendrycks2019Using, Rice2020OverfittingNetL2
from attacks import VMI_FGSM
from tester import test_transfer_attack_acc, test_acc
from data.lesson import get_lesson_loader

# test_acc(Identity(Andriushchenko2020Understanding(pretrained=True)), get_lesson_loader())
# test_acc(Identity(Sehwag2020Hydra(pretrained=True)), get_lesson_loader())
# test_acc(Identity(Carmon2019Unlabeled(pretrained=True)), get_lesson_loader())
# test_acc(Identity(Wang2020Improving(pretrained=True)), get_lesson_loader())
# test_acc(Identity(Hendrycks2019Using(pretrained=True)), get_lesson_loader())
# test_acc(Identity(Rice2020OverfittingNetL2(pretrained=True)), get_lesson_loader())
# test_acc(BaseNormModel(resnet18(pretrained=True)), get_lesson_loader())  # acc = 0 because it's an imagenet model
attacker = VMI_FGSM([
    Identity(Sehwag2020Hydra(pretrained=True)),  # model that requires normalization
    Identity(Andriushchenko2020Understanding(pretrained=True)),  # model that do not need normalization
],
total_step=20,
)

test_transfer_attack_acc(attacker,
                         get_lesson_loader(), 
                         [
                            # Identity(Sehwag2020Hydra(pretrained=True)),  # white box attack
                            # Identity(Andriushchenko2020Understanding(pretrained=True)),  # white box attack
                            Identity(Carmon2019Unlabeled(pretrained=True)),  # transfer attack
                            Identity(Wang2020Improving(pretrained=True)),  # transfer attack
                            Identity(Hendrycks2019Using(pretrained=True)),  # transfer attack
                            Identity(Rice2020OverfittingNetL2(pretrained=True)),  # transfer attack
                         ],
                         save_path='./results/lesson/images/'
                         )