from python_api.utils.class_factory import ClassFactory
from .base_nerf import BaseNeRF
from .base_nerf_sh import BaseNeRFSH
from .instant_ngp import InstantNGP


NGP_factory = ClassFactory()
NGP_factory.register('BaseNeRF', BaseNeRF)
NGP_factory.register('BaseNeRFSH', BaseNeRFSH)
NGP_factory.register('InstantNGP', InstantNGP)


__all__ = ['NGP_factory']