import torch
import torch.nn as nn

class InvariantOps(nn.Module):
    """A class with static methods to compute common tensor invariants and a custom exponential."""
    
    @staticmethod
    def cexp(x):
        # Computes exp(x) - 1
        return torch.exp(x) - 1
    
    @staticmethod
    def Invariant1(A):
        # Computes the trace of each 3x3 matrix in A.
        # A[:, 0, 0] takes the (0,0) element from every matrix in the batch.
        return A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]
    
    @staticmethod
    def Invariant2(A):
        # Computes 0.5 * ((trace(A))^2 - trace(A*A))
        B = torch.matmul(A, A)  # Multiply each 3x3 matrix with itself.
        return 0.5 * (InvariantOps.Invariant1(A)**2 - InvariantOps.Invariant1(B))
    
    @staticmethod
    def Invariant3(A):
        # Computes the determinant of each 3x3 matrix in A.
        return torch.linalg.det(A)

if __name__ == "__main__":
    # Define a batch of two 3x3 matrices.
    A = torch.tensor([
        [[1.0, 2.0, 3.0],
         [0.0, 4.0, 5.0],
         [0.0, 0.0, 6.0]],
        [[2.0, 0.0, 0.0],
         [0.0, 3.0, 0.0],
         [0.0, 0.0, 4.0]]
    ])
    
    # Print the batch A.
    print("Input A (batch of 3x3 matrices):")
    print(A)
    
    # Demonstrate the custom exponential function.
    x = torch.tensor([0.0, 1.0, 2.0])
    print("\nCustom exponential cexp(x) where x =", x)
    print(InvariantOps.cexp(x))
    
    # Compute and print the invariants for A.
    print("\nInvariant1 (trace) of each matrix in A:")
    print(InvariantOps.Invariant1(A))
    
    print("\nInvariant2 of each matrix in A:")
    print(InvariantOps.Invariant2(A))
    
    print("\nInvariant3 (determinant) of each matrix in A:")
    print(InvariantOps.Invariant3(A))
