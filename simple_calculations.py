# simple_calculations.py
# This script demonstrates some basic calculations in Python.

# --- Basic Arithmetic Operations ---
a = 10
b = 3

print("Values: a =", a, "and b =", b)

# Addition
addition = a + b
print("Addition (a + b):", addition)

# Subtraction
subtraction = a - b
print("Subtraction (a - b):", subtraction)

# Multiplication
multiplication = a * b
print("Multiplication (a * b):", multiplication)

# Division (returns a float)
division = a / b
print("Division (a / b):", division)

# Integer Division (floored division)
integer_division = a // b
print("Integer Division (a // b):", integer_division)

# Modulo (remainder)
modulo = a % b
print("Modulo (a % b):", modulo)

# Exponentiation
exponentiation = a ** b
print("Exponentiation (a ** b):", exponentiation)

# --- Factorial Calculation ---
def factorial(n):
    """Return the factorial of n computed iteratively."""
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

num = 5
print(f"Factorial of {num}:", factorial(num))

def bajs(input):
   # print(input)
    input = input + 1
  #  print(input)
    return input

print(bajs(4)) 

# --- Array Calculations with NumPy ---
import numpy as np

# Create two NumPy arrays
array1 = np.array([1, 2, 3, 4])
array2 = np.array([4, 3, 2, 1])

# Elementwise addition of arrays
array_sum = array1 + array2
print("Elementwise addition of arrays:", array_sum)
print(array_sum.T)


import torch
import torch.nn as nn
def cexp(x):
    return torch.exp(x) - 1

x = torch.tensor([1.0, 2.0, 3.0])

print(cexp(x))

class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def display(self):
        print("Name:", self.name, "Salary:", self.salary)

emp1 = Employee("Alice", 50000)
emp2 = Employee("Bob", 60000)

emp1.display()
emp2.display()
