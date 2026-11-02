"""Make a private CUDA12.8 header copy compatible with the newer host glibc."""
from pathlib import Path
import re
import shutil
root = Path(__file__).resolve().parent
headers = root / 'cuda-compat-include'
shutil.copytree('/usr/local/cuda-12.8/targets/x86_64-linux/include', headers, dirs_exist_ok=True)
f = headers / 'crt/math_functions.h'
text, count = re.subn(
    r'(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (?:double|float)\s+'
    r'(?:cospi|sinpi|rsqrt|cospif|sinpif|rsqrtf)\((?:double|float) x\));',
    r'\1 noexcept(true);', f.read_text())
assert count == 6, count
f.write_text(text)
print('Adjusted six declarations in the private header copy; installed toolkit unchanged.')
