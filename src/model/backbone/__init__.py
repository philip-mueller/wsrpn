from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from src.model.backbone.backbone_loader import BackboneConfig, Backbone

package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([package_dir]):
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute) and issubclass(attribute, Backbone) and attribute is not Backbone:
            backbone_cls = attribute
            backbone_cls.register()
