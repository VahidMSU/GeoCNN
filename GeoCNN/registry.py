import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)


from GeoCNN.losses import weighted_cross_entropy_loss
from GeoCNN.losses import (
    SpatioTemporalLoss,
    HuberLossWithThreshold,
    LogLoss,
    WeightedLogCoshLoss
)
from adabelief_pytorch import AdaBelief

# Registry containers
OPTIMIZER_REGISTRY = {}
SCHEDULER_REGISTRY = {}
LOSS_FUNCTION_REGISTRY = {}
MODEL_REGISTRY = {}
# Registry for feature retrieval functions
FEATURE_REGISTRY = {}


# Decorator for registering optimizers
def register_optimizer(name):
    def decorator(func):
        OPTIMIZER_REGISTRY[name] = func
        return func
    return decorator

# Decorator for registering schedulers
def register_scheduler(name):
    def decorator(func):
        SCHEDULER_REGISTRY[name] = func
        return func
    return decorator

# Decorator for registering loss functions
def register_loss_function(name):
    def decorator(func):
        LOSS_FUNCTION_REGISTRY[name] = func
        return func
    return decorator

def register_model(name):
    def decorator(func):
        MODEL_REGISTRY[name] = func
        return func
    return decorator

# Decorator for registering feature retrieval functions
def register_feature_retriever(name):
    def decorator(func):
        FEATURE_REGISTRY[name] = func
        return func
    return decorator

# ------------------- Feature Retrieval Registration -------------------
# Registering the get_var_name function
@register_feature_retriever("default_feature_retriever")
def get_var_name(feature_type, RESOLUTION):
    if feature_type == 'numerical':
        return [
            f"LC20_Asp_220_{RESOLUTION}m",
            f"LC20_BPS_220_{RESOLUTION}m",
            f"LC20_EVT_220_{RESOLUTION}m",
            f"LC20_Elev_220_{RESOLUTION}m",
            f"LC20_SlpD_220_{RESOLUTION}m",
            f"LC20_SlpP_220_{RESOLUTION}m",
            "gssurgo/soil_k",
            "gssurgo/dp",
            "gssurgo/bd",
            "gssurgo/awc",
            "gssurgo/carbon",
            "gssurgo/clay",
            "gssurgo/silt",
            "gssurgo/sand",
            "gssurgo/rock",
            "gssurgo/alb",
        ]
    elif feature_type == 'categorical':
        return [
            f'landuse_{RESOLUTION}m',
        ]
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    


# ------------------- Optimizer Registration -------------------
@register_optimizer("AdamW")
def adamw_optimizer(model, config):
    return AdamW(model.parameters(), lr=config['opt_lr'], weight_decay=config['weight_decay'])

@register_optimizer("AdaBelief")
def adabelief_optimizer(model, config):
    return AdaBelief(
        model.parameters(),
        lr=config['opt_lr'],
        eps=1e-16,
        betas=(0.9, 0.999),
        weight_decay=config['weight_decay'],
        weight_decouple=True,
        rectify=True
    )

@register_optimizer("Adam")
def adam_optimizer(model, config):
    return Adam(model.parameters(), lr=config['opt_lr'], weight_decay=config['weight_decay'])

# ------------------- Scheduler Registration -------------------
@register_scheduler("CosineAnnealingLR")
def cosine_annealing_scheduler(optimizer, config, total_effective_steps):
    return CosineAnnealingLR(optimizer, T_max=total_effective_steps, eta_min=0)

@register_scheduler("ReduceLROnPlateau")
def reduce_lr_on_plateau_scheduler(optimizer, config, total_effective_steps):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

@register_scheduler("Linear")
def linear_scheduler(optimizer, config, total_effective_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_effective_steps // 100,
        num_training_epochs=total_effective_steps 
    )

@register_scheduler("CosineAnnealingHardRestarts")
def cosine_hard_restarts_scheduler(optimizer, config, total_effective_steps):
    from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
    """
    Configures a cosine annealing scheduler with hard restarts and warmup.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        config (dict): Configuration containing optional 'num_cycles'.
        total_effective_steps (int): Total number of optimizer steps across all epochs.
    Returns:
        torch.optim.lr_scheduler: A cosine annealing scheduler with hard restarts and warmup.
    """
    # Calculate warmup steps as 10% of total steps, with a minimum of 1
    num_warmup_steps = max(1, int(0.10 * total_effective_steps))

    # Get number of cycles for hard restarts; default is 1 if not provided
    num_cycles = config.get('num_cycles', 1)

    # Log configuration for debugging
    print("Scheduler Configuration:")
    print(f"  Total Effective Steps: {total_effective_steps}")
    print(f"  Warmup Steps: {num_warmup_steps}")
    print(f"  Number of Cycles: {num_cycles}")

    return get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_epochs=total_effective_steps,
        num_cycles=num_cycles,
    )


# ------------------- Loss Function Registration -------------------
@register_loss_function("MSELoss")
def mse_loss():
    return torch.nn.MSELoss()

@register_loss_function("L1Loss")
def l1_loss():
    return torch.nn.L1Loss()

@register_loss_function("SmoothL1Loss")
def smooth_l1_loss():
    return torch.nn.SmoothL1Loss()

@register_loss_function("CrossEntropyLoss")
def cross_entropy_loss():
    return torch.nn.CrossEntropyLoss()

@register_loss_function("WeightedCrossEntropyLoss")
def weighted_cross_entropy(config, device):
    if device is None:
        raise ValueError("Device must be specified for WeightedCrossEntropyLoss")
    return weighted_cross_entropy_loss(num_classes=config['num_classes'], zero_weight=0.01, device=device)

@register_loss_function("BCELoss")
def bce_loss():
    return torch.nn.BCELoss()

@register_loss_function("BCEWithLogitsLoss")
def bce_with_logits_loss():
    return torch.nn.BCEWithLogitsLoss()

@register_loss_function("SpatioTemporalLoss")
def spatio_temporal_loss(config):
    return SpatioTemporalLoss(config)

@register_loss_function("HuberLossWithThreshold")
def huber_loss_with_threshold():
    return HuberLossWithThreshold()

@register_loss_function("LogLoss")
def log_loss():
    return LogLoss()

@register_loss_function("WeightedLogCoshLoss")
def weighted_log_cosh_loss():
    return WeightedLogCoshLoss()


# ------------------- Model Registration -------------------
@register_model("ResNetUnet")
def resnet_unet(config, device):
    from GeoCNN.GeoCNN_models.ResNetUnet import ResNetUnet
    return ResNetUnet(num_channels=config['num_channels'], extra_blocks=1, pretrained=True).to(device)

@register_model("ResNet101Transformer")
def resnet_transformer(config, device):
    from GeoCNN.GeoCNN_models.ResNet101Transformer import ResNet101Transformer
    return ResNet101Transformer(num_channels=config['num_channels'], 
                                embed_dim=config['embed_dim'],
                                num_heads=config['num_heads'],
                                forward_expansion=config['forward_expansion'],
                                num_layers=config['num_layers'],
                                dropout=config['dropout'],
                                pretrained=False, 
                                num_classes=config['num_classes']).to(device)


@register_model("CNNTransformerRegressor_v8")
def cnn_transformer_v8(config, device):
    ## best model so far
    from GeoCNN.GeoCNN_models.CNNTransformerRegressor_v8 import CNNTransformerRegressor_v8
    return CNNTransformerRegressor_v8(
        num_channels=config['num_channels'],
        num_categorical_channels=config['num_categorical_channels'],
        num_dynamic_channels=config['num_dynamic_channels'],
        num_static_channels=config['num_static_channels'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        forward_expansion=config['forward_expansion'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_classes=config['num_classes'],
    ).to(device)


@register_model("CNNTransformerRegressor_v11")
def cnn_transformer_v11(config, device):
    ### similar to V8 but static and dynamic channels are separated
    from GeoCNN.GeoCNN_models.CNNTransformerRegressor_v11 import CNNTransformerRegressor_v11
    return CNNTransformerRegressor_v11(
        num_static_channels=config['num_static_channels'],
        num_dynamic_channels=config['num_dynamic_channels'],
        num_categorical_channels=config['num_categorical_channels'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        forward_expansion=config['forward_expansion'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_classes=1,
    ).to(device)

@register_model("CNNTransformerRegressor_v12")
def cnn_transformer_v12(config, device):
    ### similar to V8 but with several layer normalization
    from GeoCNN.GeoCNN_models.CNNTransformerRegressor_v12 import CNNTransformerRegressor_v12
    return CNNTransformerRegressor_v12(
        num_channels=config['num_channels'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        forward_expansion=config['forward_expansion'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_classes=config['num_classes'],
    ).to(device)


@register_model("ForcedSTRNN")
def forced_strnn(config, device):
    from GeoCNN.GeoCNN_models.ForcedSTRNN import ForcedSTRNN
    return ForcedSTRNN(num_layers=2, num_hidden=[32, 32], 
                        img_channel=1, act_channel=9, 
                        init_cond_channel=1, 
                        static_channel=config['num_static_channels'],
                        out_channel=1).to(device)

@register_model("MultiScaleXception")
def multi_scale_xception(config, device):
    from GeoCNN.GeoCNN_models.MultiScaleXception import MultiScaleXception
    return MultiScaleXception(num_channels=config['num_channels'],
                                output_channels=1,
                                dropout_rate=config['dropout'],
                                batch_window=config['batch_window'],
                                
                                ).to(device)

@register_model("VisionTransformerForRegression")
def vision_transformer_for_regression(config, device):
    from GeoCNN.GeoCNN_models.VisionTransformerForRegression import VisionTransformerForRegression
    return VisionTransformerForRegression(in_channels=config['num_channels'],
                                                  embed_dim=config['embed_dim'],
                                                    num_heads=config['num_heads'],
                                                    mlp_dim=config['mlp_dim'],
                                                    depth=config['depth'],
                                                    dropout=config['dropout'],
                                                    batch_window=config['batch_window'],
                                                    num_classes=config['num_classes']
                                                    ).to(device)

@register_model("VisionTransformer")
def vision_transformer(config, device):
    from GeoCNN.GeoCNN_models.VisionTransformer import VisionTransformer
    return VisionTransformer(in_channels=config['num_channels'],
                             embed_dim=config['embed_dim'],
                             num_heads=config['num_heads'],
                             depth=config['depth'],
                             mlp_dim=config['mlp_dim'],
                             num_classes=config['num_classes']
                             ).to(device)

# ------------------- Helper Functions -------------------
def setup_optimizer(config, model):
    optimizer_name = config['optimizer']
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer {optimizer_name} not recognized")
    return OPTIMIZER_REGISTRY[optimizer_name](model, config)

def select_scheduler(optimizer, config, total_effective_steps):
    print(f"{total_effective_steps} effective training steps")
    scheduler_name = config['scheduler']
    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler {scheduler_name} not recognized")
    return SCHEDULER_REGISTRY[scheduler_name](optimizer, config, total_effective_steps)

def select_criterion(config, device=None):
    loss_name = config['loss_function']
    if loss_name not in LOSS_FUNCTION_REGISTRY:
        raise ValueError(f"Loss function {loss_name} not recognized")
    if loss_name == "WeightedCrossEntropyLoss":
        return LOSS_FUNCTION_REGISTRY[loss_name](config, device)
    elif loss_name == "SpatioTemporalLoss":
        return LOSS_FUNCTION_REGISTRY[loss_name](config)
    else:
        return LOSS_FUNCTION_REGISTRY[loss_name]()


def retrieve_features(name, feature_type, RESOLUTION):
    if name not in FEATURE_REGISTRY:
        raise ValueError(f"Feature retriever {name} not found in registry")
    return FEATURE_REGISTRY[name](feature_type, RESOLUTION)

def select_model(config, device):
    model_name = config['model']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not recognized")
    return MODEL_REGISTRY[model_name](config, device)
