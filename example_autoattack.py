'''
Average transfer attack accuracy:  0.808
Average SSIM:  0.9514
Score:  0.7687
----------------------------------------------------------------------------------------------------
<class 'models.BaseNormModel.Identity'> <class 'robustbench.model_zoo.architectures.wide_resnet.WideResNet'> 0.91
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
<class 'models.BaseNormModel.Identity'> <class 'robustbench.model_zoo.architectures.wide_resnet.WideResNet'> 0.848
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
<class 'models.BaseNormModel.Identity'> <class 'robustbench.model_zoo.cifar10.Hendrycks2019UsingNet'> 0.87
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
<class 'models.BaseNormModel.Identity'> <class 'robustbench.model_zoo.cifar10.Rice2020OverfittingNetL2'> 0.604
----------------------------------------------------------------------------------------------------
对抗样本已保存到: ./results/lesson/images_autoattack_test_1/
'''
from models.BaseNormModel import Identity
from models.RobustBench.cifar10 import (
    # Linf模型
    Andriushchenko2020Understanding, 
    Carmon2019Unlabeled, 
    Sehwag2020Hydra, 
    Wang2020Improving, 
    Hendrycks2019Using,
    Bartoldson2024Adversarial_WRN_94_16,
    Amini2024MeanSparse_S_WRN_94_16,
    Peng2023Robust,
    Bai2024MixedNUTS,
    # L2模型
    Rice2020OverfittingNetL2,
    Rebuffi2021Fixing_70_16_cutmix_extra_L2,
    Gowal2020Uncovering_L2,
    Augustin2020Adversarial_34_10_L2,
    Engstrom2019Robustness_L2,
    Wu2020Adversarial_L2,
    Ding2020MMA_L2,
    Wang2023Better_WRN_70_16,
    Amini2024MeanSparse_S_WRN_70_16,
)
from tester import test_transfer_attack_acc
from data.lesson import get_lesson_loader
from attacks.autoattack import AutoAttack

import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        # 平均化所有模型的输出
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output


class AutoAttackWrapper:
    def __init__(self, attack, batch_size=64):
        self.attack = attack
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
    
    def __call__(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        return self.attack.run_standard_evaluation(x, y, bs=self.batch_size, return_labels=False)


# 配置参数
BATCH_SIZE = 16  # AutoAttack的batch size（如果内存不足，可以减少到16或8）
USE_NORM = 'L2'  # L2 或 Linf 或 All
ATTACK_NORM = 'Linf'  # 攻击norm类型：'Linf' 或 'L2'（当使用混合模型时，建议使用L2）
GPU_ID = 3  # 指定使用的GPU编号

# 选择GPU设备
if torch.cuda.is_available():
    if GPU_ID >= torch.cuda.device_count():
        print(f"警告: GPU {GPU_ID} 不存在，使用 GPU 0")
        GPU_ID = 0
    device = torch.device(f'cuda:{GPU_ID}')
    torch.cuda.set_device(GPU_ID)
    print(f"使用设备: {device}")
    print(f"GPU 名称: {torch.cuda.get_device_name(GPU_ID)}")
    gpu_props = torch.cuda.get_device_properties(GPU_ID)
    print(f"GPU 总内存: {gpu_props.total_memory / 1024**3:.2f} GB")
    print(f"GPU 初始内存使用: {torch.cuda.memory_allocated(GPU_ID) / 1024**3:.2f} GB")
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print(f"使用设备: {device}")
print()

# 加载集成模型（用于生成攻击）- 同时使用L2和Linf模型
if USE_NORM == 'All':
    # Linf模型列表
    linf_models = [
        Identity(Sehwag2020Hydra(pretrained=True)),  # Linf模型
        Identity(Andriushchenko2020Understanding(pretrained=True)),  # Linf模型
        Identity(Bartoldson2024Adversarial_WRN_94_16(pretrained=True)),  # Linf模型
        Identity(Amini2024MeanSparse_S_WRN_94_16(pretrained=True)),  # Linf模型
        Identity(Peng2023Robust(pretrained=True)),  # Linf模型
        # Identity(Bai2024MixedNUTS(pretrained=True)),  # remove for too slow, 好像架构不一样，建议别用
        # Identity(Carmon2019Unlabeled(pretrained=True)),  # Linf模型
        # Identity(Wang2020Improving(pretrained=True)),  # Linf模型
        # Identity(Hendrycks2019Using(pretrained=True)),  # Linf模型
    ]
    
    # L2模型列表
    l2_models = [
        # 顶级L2模型（性能最强）
        Identity(Wang2023Better_WRN_70_16(pretrained=True)),
        Identity(Amini2024MeanSparse_S_WRN_70_16(pretrained=True)),
        Identity(Rebuffi2021Fixing_70_16_cutmix_extra_L2(pretrained=True)),  # 2021年最佳L2模型
        Identity(Gowal2020Uncovering_L2(pretrained=True)),  # DeepMind顶级L2模型
        Identity(Augustin2020Adversarial_34_10_L2(pretrained=True)),  # 顶级L2模型
        # Identity(Engstrom2019Robustness_L2(pretrained=True)),  # 经典L2鲁棒模型
        # 强L2模型
        # Identity(Wu2020Adversarial_L2(pretrained=True)),  # 强L2模型
        # Identity(Ding2020MMA_L2(pretrained=True)),  # MMA训练方法
        # Identity(Rice2020OverfittingNetL2(pretrained=True)),  # L2模型
    ]
    
    # 选择指定数量的Linf和L2模型
    selected_linf = linf_models[:]  # NUM_LINF_MODELS]
    selected_l2 = l2_models[:] # NUM_L2_MODELS]
    ensemble_models = selected_linf + selected_l2
    
    print("集成模型配置（混合使用L2和Linf模型）:")
    print(f"  Linf模型: {len(selected_linf)} 个")
    for i, model in enumerate(selected_linf, 1):
        inner_name = type(model.model).__name__ if hasattr(model, 'model') else type(model).__name__
        print(f"    [{i}] {inner_name} (Linf)")
    print(f"  L2模型: {len(selected_l2)} 个")
    for i, model in enumerate(selected_l2, 1):
        inner_name = type(model.model).__name__ if hasattr(model, 'model') else type(model).__name__
        print(f"    [{i}] {inner_name} (L2)")
    print()
elif USE_NORM == 'Linf':
    # 只使用Linf模型（原始逻辑）
    all_available_models = [
        Identity(Sehwag2020Hydra(pretrained=True)),
        Identity(Andriushchenko2020Understanding(pretrained=True)),
        Identity(Bartoldson2024Adversarial_WRN_94_16(pretrained=True)),
        Identity(Amini2024MeanSparse_S_WRN_94_16(pretrained=True)),
        Identity(Peng2023Robust(pretrained=True)),
        # Identity(Bai2024MixedNUTS(pretrained=True)),  # remove for too slow, 好像架构不一样，建议别用
        # Identity(Carmon2019Unlabeled(pretrained=True)),
        # Identity(Wang2020Improving(pretrained=True)),
        # Identity(Hendrycks2019Using(pretrained=True)),
    ]
    ensemble_models = all_available_models[:] # NUM_LINF_MODELS]
    print("集成模型配置（仅使用Linf模型）:")
    print(f"  Linf模型: {len(ensemble_models)} 个\n")
elif USE_NORM == 'L2':
    # 只使用L2模型
    all_available_models = [
        # 顶级L2模型（性能最强）
        Identity(Wang2023Better_WRN_70_16(pretrained=True)),
        Identity(Amini2024MeanSparse_S_WRN_70_16(pretrained=True)),
        Identity(Rebuffi2021Fixing_70_16_cutmix_extra_L2(pretrained=True)),  # 2021年最佳L2模型
        Identity(Gowal2020Uncovering_L2(pretrained=True)),  # DeepMind顶级L2模型
        Identity(Augustin2020Adversarial_34_10_L2(pretrained=True)),  # 顶级L2模型
        # Identity(Engstrom2019Robustness_L2(pretrained=True)),  # 经典L2鲁棒模型
        # 强L2模型
        # Identity(Wu2020Adversarial_L2(pretrained=True)),  # 强L2模型
        # Identity(Ding2020MMA_L2(pretrained=True)),  # MMA训练方法
        # Identity(Rice2020OverfittingNetL2(pretrained=True)),  # L2模型
    ]
    ensemble_models = all_available_models[:] # NUM_L2_MODELS]
    print("集成模型配置（仅使用L2模型）:")
    print(f"  L2模型: {len(ensemble_models)} 个\n")
else:
    raise ValueError(f"不支持的norm类型: {USE_NORM}")

# 初始化模型并创建集成模型
print("正在加载模型到GPU...")
for i, model in enumerate(ensemble_models, 1):
    model.to(device).eval()
    inner_name = type(model.model).__name__ if hasattr(model, 'model') else type(model).__name__
    print(f"  [{i}/{len(ensemble_models)}] {inner_name} - 已加载")

ensemble = EnsembleModel(ensemble_models).to(device).eval()
print(f"\n已加载 {len(ensemble_models)} 个模型并创建集成模型\n")

# 创建AutoAttack攻击器
# 注意：当使用混合模型时，建议使用L2 norm（因为L2 norm通常更通用）
# 如果只使用Linf模型，可以使用Linf norm（eps=8/255）
if USE_NORM == 'All':
    attack_eps = 2.0  # L2 norm的eps
    attack_norm = ATTACK_NORM
    print(f"使用混合模型（L2+Linf），攻击norm: {attack_norm}, eps: {attack_eps}")
elif USE_NORM == 'Linf':
    # 根据攻击norm选择eps
    if ATTACK_NORM == 'Linf':
        attack_eps = 8/255
    else:
        attack_eps = 2.0
    attack_norm = ATTACK_NORM
    print(f"使用Linf模型，攻击norm: {attack_norm}, eps: {attack_eps:.6f}")
elif USE_NORM == 'L2':
    # 根据攻击norm选择eps
    if ATTACK_NORM == 'L2':
        attack_eps = 2.0
    else:
        attack_eps = 8/255
    attack_norm = ATTACK_NORM
    print(f"使用L2模型，攻击norm: {attack_norm}, eps: {attack_eps:.6f}")

attack = AutoAttack(
    ensemble,
    norm=attack_norm,
    eps=attack_eps,
    version='plus',  # 可选: 'standard', 'plus', 'rand'
    verbose=True,
    device=device
)
print()
attacker = AutoAttackWrapper(attack, batch_size=BATCH_SIZE)

# 指定要测试的目标模型（直接使用，不进行过滤）
target_models = [
    Identity(Carmon2019Unlabeled(pretrained=True)),
    Identity(Wang2020Improving(pretrained=True)),
    Identity(Hendrycks2019Using(pretrained=True)),
    Identity(Rice2020OverfittingNetL2(pretrained=True)),
]

# 显示将要测试的模型信息
print(f"将测试 {len(target_models)} 个目标模型（迁移攻击）:")
for i, model in enumerate(target_models, 1):
    if hasattr(model, 'model'):
        inner_name = type(model.model).__name__
        print(f"  [{i}] {inner_name}")
    else:
        print(f"  [{i}] {type(model).__name__}")
print()

# 将目标模型移动到GPU
for model in target_models:
    model.to(device).eval()

print(f"开始测试 {len(target_models)} 个目标模型（AutoAttack可能需要几分钟）...\n")

test_transfer_attack_acc(
    attacker,
    get_lesson_loader(), 
    target_models,
    device=device,  # 显式传递device参数确保使用GPU
    save_path='./results/lesson/images_autoattack_test/'
)
print("对抗样本已保存到: ./results/lesson/images_autoattack_test/")
