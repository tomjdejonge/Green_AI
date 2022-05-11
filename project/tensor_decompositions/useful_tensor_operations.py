import numpy as np
import tensorly as tl
from tensorly.testing import assert_array_equal

# Creating a (5x2x4) 3 way-tensor with tensorly as a numpy array
orig_tensor = tl.tensor(np.arange(40).reshape((5, 2, 4)))

# Displaying tensor
#print('Original tensor:' + 2*'\n' + f'{orig_tensor}')

# Unfolding tensor
# for m in range(orig_tensor.ndim):
#     print(f'Mode-{m} unfolding: \n{tl.unfold(orig_tensor, m)}')

# Refolding the tensor
# for mode in range(orig_tensor.ndim):
#     unf = tl.unfold(orig_tensor, mode) # Unfold original tensor
#     refold = tl.fold(unf, mode, orig_tensor.shape) # Refold tensor
#     assert_array_equal(refold, orig_tensor) # Check equality

A = np.array([[1, 2, 3], [3, 4, 5]])
print(A.shape)


