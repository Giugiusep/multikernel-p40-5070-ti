"""Read-only CUDA-driver enumeration; no model or context allocation."""
import ctypes,json
lib=ctypes.CDLL('libcuda.so.1')
r={'cuInit':lib.cuInit(0)}
if r['cuInit']==0:
 count=ctypes.c_int();r['cuDeviceGetCount']=lib.cuDeviceGetCount(ctypes.byref(count));r['devices']=[]
 for i in range(count.value):
  dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),i)==0
  name=ctypes.create_string_buffer(256);assert lib.cuDeviceGetName(name,256,dev)==0
  r['devices'].append(name.value.decode())
print(json.dumps(r,indent=2))
