from python_api.utils.class_factory import ClassFactory
from .pixel_loss import PixelLoss, PixelLossWithDepth
from .pixel_loss import PixelLossWithDepthAndSight
from .pixel_loss import PixelLossWithSightAndNear


Loss_factory = ClassFactory()
Loss_factory.register('PixelLoss', PixelLoss)
Loss_factory.register('PixelLossWithDepth', PixelLossWithDepth)
Loss_factory.register('PixelLossWithDepthAndSight', PixelLossWithDepthAndSight)
Loss_factory.register('PixelLossWithSightAndNear', PixelLossWithSightAndNear)


__all__ = ['Loss_factory']
