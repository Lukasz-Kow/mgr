"""Init file for models module."""

from .backbone import ResNetBackbone2D, ResNet3DBackbone, get_backbone
from .baseline_softmax import BaselineSoftmaxModel, BaselineTrainer
from .selective_net import SelectiveNet, SelectiveNetLoss
from .evidential_layer import EvidentialLayer, EvidentialLoss, compute_uncertainty
from .hybrid_model import HybridEvidentialModel

__all__ = [
    # Backbones
    'ResNetBackbone2D',
    'ResNet3DBackbone',
    'get_backbone',
    
    # Baseline
    'BaselineSoftmaxModel',
    'BaselineTrainer',
    
    # SelectiveNet
    'SelectiveNet',
    'SelectiveNetLoss',
    
    # Evidential
    'EvidentialLayer',
    'EvidentialLoss',
    'compute_uncertainty',
    
    # Hybrid
    'HybridEvidentialModel',
]
