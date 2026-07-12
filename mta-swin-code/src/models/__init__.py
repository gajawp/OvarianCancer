from .mta_swin import (
    MTASwin,
    create_mta_swin,
    create_mta_swin_tiny,
    create_mta_swin_small,
    create_mta_swin_base,
    create_mta_swin_large
)

MODEL_REGISTRY = {
    'mta_swin_tiny': create_mta_swin_tiny,
    'mta_swin_small': create_mta_swin_small,
    'mta_swin_base': create_mta_swin_base,
    'mta_swin_large': create_mta_swin_large,
}

def create_model(model_name, **kwargs):
    """
    Create model from config parameters.
    
    Args:
        model_name (str): Model name from registry
        **kwargs: All model parameters from config
    
    Returns:
        torch.nn.Module: Created model
    """
    if model_name not in MODEL_REGISTRY:
        available_models = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(**kwargs)

# Export
__all__ = [
    'MTASwin',
    'create_model', 
    'create_mta_swin',
    'create_mta_swin_tiny',
    'create_mta_swin_small', 
    'create_mta_swin_base',
    'create_mta_swin_large',
    'MODEL_REGISTRY'
]
