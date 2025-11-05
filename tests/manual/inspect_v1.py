import importlib
import v1
print('v1 namespace', v1.__path__)
mod = importlib.import_module('v1.attention.backends.ring_flash_attn')
print('import ok', mod)
