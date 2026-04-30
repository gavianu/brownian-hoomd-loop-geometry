import cupy as cp
print("GPU count:", cp.cuda.runtime.getDeviceCount())
x = cp.arange(10, dtype=cp.float32)
y = cp.sin(x)            # declanșează compilare NVRTC
cp.cuda.Stream.null.synchronize()
print("OK, sin compiled on:", y.device)
print("CUDA runtime version:", cp.cuda.runtime.runtimeGetVersion())
print("Driver version:", cp.cuda.runtime.driverGetVersion())

# import cupy as cp, sys
# print("Python:", sys.executable)
# print("CuPy file:", cp.__file__)
# print("GPU count:", cp.cuda.runtime.getDeviceCount())
# x = cp.arange(10, dtype=cp.float32)    # declanșează NVRTC
# print("OK on device:", x.device)
# print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
# print("Driver:", cp.cuda.runtime.driverGetVersion())

# import os, inspect
# import cupy_backends.cuda.libs as L
# libs = os.path.dirname(inspect.getfile(L))
# print("libs dir:", libs)
# print("NVRTC present?:", [f for f in os.listdir(libs) if f.lower().startswith("nvrtc")])
