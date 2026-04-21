"""
Trainer模块
封装训练逻辑
"""

from .ps_family_trainer import PS_Family_Trainer
# from .rome_trainer import RoMETrainer

__all__ = ['PS_Family_Trainer', 'RoMETrainer']
