# import cupy as cp

# print(cp.cuda.runtime.getDeviceCount())
# print(cp.cuda.Device(0).name)
# cp.show_config()

import cupy as cp

print(cp.cuda.runtime.getDeviceCount())
print(cp.cuda.Device(0))
cp.show_config()