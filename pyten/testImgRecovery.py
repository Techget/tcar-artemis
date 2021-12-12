import matplotlib.image as mpimg  # Use it to load image
import numpy as np

lena = mpimg.imread("./test/testImg.png")
im = np.double(np.uint8(lena * 255))
im = im[0:50, 0:50, 0:3]

from pyten.tenclass import Tensor  # Use it to construct Tensor object

X1 = Tensor(im)  # Construct Image Tensor to be Completed
X0 = X1.data.copy()
X0 = Tensor(X0)  # Save the Ground Truth
Omega1 = (im < 100) * 1.0  # Missing index Tensor
X1.data[Omega1 == 0] = 0

# Basic Tensor Completion with methods: CP-ALS, Tucker-ALS, FaLRTC, SiLRTC, HaLRTC, TNCP
from pyten.method import *

r = 10
R = [10, 10, 3]  # Rank for tucker-based methods
[T1, rX1] = cp_als(X1, r, Omega1, maxiter=1000, printitn=100)
[T2, rX2] = tucker_als(X1, R, Omega1, max_iter=100, printitn=100)
alpha = np.array([1.0, 1.0, 1e-3])
alpha = alpha / sum(alpha)
rX3 = falrtc(X1, Omega1, max_iter=100, alpha=alpha)
rX4 = silrtc(X1, Omega1, max_iter=100, alpha=alpha)
rX5 = halrtc(X1, Omega1, max_iter=100, alpha=alpha)
self1 = TNCP(X1, Omega1, rank=r)
self1.run()

# Error Testing
from pyten.tools import tenerror

realX = X0
[Err1, ReErr11, ReErr21] = tenerror(rX1, realX, Omega1)
[Err2, ReErr12, ReErr22] = tenerror(rX2, realX, Omega1)
[Err3, ReErr13, ReErr23] = tenerror(rX3, realX, Omega1)
[Err4, ReErr14, ReErr24] = tenerror(rX4, realX, Omega1)
[Err5, ReErr15, ReErr25] = tenerror(rX5, realX, Omega1)
[Err6, ReErr16, ReErr26] = tenerror(self1.X, realX, Omega1)
print ('\n', 'The Relative Error of the Six Methods are:', ReErr21, ReErr22, ReErr23, ReErr24, ReErr25, ReErr26)