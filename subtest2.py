import torch
import torch.nn as nn

class PowerExpansionConvex(nn.Module):
    """
    This class takes a tensor x (which could be a batch of scalars)
    and returns a tensor with several "power" terms. For each element in x,
    it returns a vector containing [x^1, x^2, x^4, x^6, ..., x^(2*n)].
    
    Note: It starts with x^1 and then uses even exponents: 2, 4, 6, ... (2*n).
    """
    def __init__(self, n):
        super().__init__()       # Initialize the base class, nn.Module.
        self.n = n               # Store the number of additional power terms to compute.
                                  # For example, if n = 3, the output will have 1 + 3 = 4 columns.

    def forward(self, x):
        """
        The forward method defines how the module processes input.
        
        Parameters:
        x: A tensor with shape [batch_size] or [batch_size, 1].
        
        Returns:
        A tensor of shape [batch_size, n+1] containing:
        [x^1, x^(2*1), x^(2*2), ..., x^(2*n)] for each element.
        """
        # Reshape x into a 1D tensor with shape [batch_size]. The view(-1) function flattens x.
        print("Input x:", x)
        x = x.view(-1)
        print("Input x:", x)

        
        # Create a list that will hold all the power terms. Start with x^1 (which is just x).
        powers = [x]
        print("Powers:", powers)
        
        # Loop from 1 to n (inclusive) to compute even powers of x.
        # For k = 1, compute x^(2*1) which is x^2.
        # For k = 2, compute x^(2*2) which is x^4, and so on.
        for k in range(1, self.n + 1):
            powers.append(x ** (2 * k))
            print("Powers:", powers)
        
        # Use torch.stack to combine the list of tensors into one tensor along a new dimension.
        # This new dimension (dim=1) will hold the power terms for each batch element.
        return torch.stack(powers, dim=1)

if __name__ == "__main__":
    # Example usage of PowerExpansionConvex:
    
    # Create a tensor representing a batch of scalar inputs.
    # For example, a batch of 3 scalars.
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Create an instance of PowerExpansionConvex with n = 3.
    # This means it will compute x^1, x^2, x^4, and x^6 (a total of 4 terms).
    expander = PowerExpansionConvex(n=3)
    
    # Run the forward pass: Pass x through the expander.
    output = expander(x)
    
    # Print the input and the expanded output.
    print("Input x:", x)
    print("Output power expansion:")
    print(output)
    
