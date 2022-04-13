from python_api.utils.class_factory import ClassFactory
from .pixel_loss import PixelLoss, PixelLossWithDepth


Loss_factory = ClassFactory()
Loss_factory.register('PixelLoss', PixelLoss)
Loss_factory.register('PixelLossWithDepth', PixelLossWithDepth)


__all__ = ['Loss_factory']
