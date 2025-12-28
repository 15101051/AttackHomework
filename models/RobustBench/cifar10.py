from robustbench.utils import load_model
from robustbench.model_zoo.cifar10 import cifar_10_models

model_dir = "/mnt/data/wenjiahao/attack/models"


def Andriushchenko2020Understanding(pretrained=True):
    return load_model(model_name="Andriushchenko2020Understanding", dataset="cifar10", threat_model="Linf", model_dir=model_dir)


def Carmon2019Unlabeled(pretrained=True):
    return load_model(model_name="Carmon2019Unlabeled", dataset="cifar10", threat_model="Linf", model_dir=model_dir)


def Sehwag2020Hydra(pretrained=True):
    return load_model(model_name="Sehwag2020Hydra", dataset="cifar10", threat_model="Linf", model_dir=model_dir)


def Wang2020Improving(pretrained=True):
    return load_model(model_name="Wang2020Improving", dataset="cifar10", threat_model="Linf", model_dir=model_dir)


def Hendrycks2019Using(pretrained=True):
    return load_model(model_name="Hendrycks2019Using", dataset="cifar10", threat_model="Linf", model_dir=model_dir)


def Bartoldson2024Adversarial_WRN_94_16(pretrained=True):
    return load_model(model_name="Bartoldson2024Adversarial_WRN-94-16", dataset="cifar10", threat_model="Linf", model_dir=model_dir)


def Amini2024MeanSparse_S_WRN_94_16(pretrained=True):
    return load_model(model_name="Amini2024MeanSparse_S-WRN-94-16", dataset="cifar10", threat_model="Linf", model_dir=model_dir)


def Peng2023Robust(pretrained=True):
    return load_model(model_name="Peng2023Robust", dataset="cifar10", threat_model="Linf", model_dir=model_dir)


def Bai2024MixedNUTS(pretrained=True):
    return load_model(model_name="Bai2024MixedNUTS", dataset="cifar10", threat_model="Linf", model_dir=model_dir)

################## Below Are L2 Models ##########################################

def Rice2020OverfittingNetL2(pretrained=True):
    return load_model(model_name="Rice2020Overfitting", dataset="cifar10", threat_model="L2", model_dir=model_dir)


def Rebuffi2021Fixing_70_16_cutmix_extra_L2(pretrained=True):
    return load_model(model_name="Rebuffi2021Fixing_70_16_cutmix_extra", dataset="cifar10", threat_model="L2", model_dir=model_dir)


def Gowal2020Uncovering_L2(pretrained=True):
    return load_model(model_name="Gowal2020Uncovering", dataset="cifar10", threat_model="L2", model_dir=model_dir)


def Augustin2020Adversarial_34_10_L2(pretrained=True):
    return load_model(model_name="Augustin2020Adversarial_34_10", dataset="cifar10", threat_model="L2", model_dir=model_dir)


def Engstrom2019Robustness_L2(pretrained=True):
    return load_model(model_name="Engstrom2019Robustness", dataset="cifar10", threat_model="L2", model_dir=model_dir)


def Wu2020Adversarial_L2(pretrained=True):
    return load_model(model_name="Wu2020Adversarial", dataset="cifar10", threat_model="L2", model_dir=model_dir)


def Ding2020MMA_L2(pretrained=True):
    return load_model(model_name="Ding2020MMA", dataset="cifar10", threat_model="L2", model_dir=model_dir)


def Wang2023Better_WRN_70_16(pretrained=True):
    return load_model(model_name="Wang2023Better_WRN-70-16", dataset="cifar10", threat_model="L2", model_dir=model_dir)


def Amini2024MeanSparse_S_WRN_70_16(pretrained=True):
    return load_model(model_name="Amini2024MeanSparse_S-WRN-70-16", dataset="cifar10", threat_model="L2", model_dir=model_dir)
