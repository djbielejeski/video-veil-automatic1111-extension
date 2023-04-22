import importlib

controlnet_external_code = None
controlnet_global_state = None
controlnet_preprocessors = None
reverse_preprocessor_aliases = None
controlnet_HWC3 = None
ControlNetUnit = None
try:
    controlnet_external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
    ControlNetUnit = controlnet_external_code.ControlNetUnit

    controlnet_annotator_util = importlib.import_module('extensions.sd-webui-controlnet.annotator.util', 'util')
    controlnet_HWC3 = controlnet_annotator_util.HWC3

    controlnet_global_state = importlib.import_module('extensions.sd-webui-controlnet.scripts.global_state', 'global_state')
    reverse_preprocessor_aliases = controlnet_global_state.reverse_preprocessor_aliases
    controlnet_preprocessors = controlnet_global_state.cn_preprocessor_modules

except ImportError:
    pass
