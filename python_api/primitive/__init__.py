from python_api.utils.class_factory import ClassFactory
from .base_nerf import BaseNeRF
from .base_nerf_sh import BaseNeRFSH
from .instant_ngp import InstantNGP
from .instant_ngp_mock import InstantNGPMock, InstantNGPMock2
from .neus import NeuS


NGP_factory = ClassFactory()
NGP_factory.register('BaseNeRF', BaseNeRF)
NGP_factory.register('BaseNeRFSH', BaseNeRFSH)
NGP_factory.register('InstantNGP', InstantNGP)
NGP_factory.register('InstantNGPMock', InstantNGPMock)
NGP_factory.register('InstantNGPMock2', InstantNGPMock2)
NGP_factory.register('NeuS', NeuS)


__all__ = ['NGP_factory']