# Python Programming Fundamentals

## Introduction to Python
Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python has become one of the most popular programming languages in the world.

## Key Features
- **Easy to Learn**: Python has a simple syntax that is easy to read and write
- **Interpreted Language**: Python code is executed line by line
- **Dynamically Typed**: Variable types are determined at runtime
- **Object-Oriented**: Supports OOP principles
- **Rich Standard Library**: Comes with extensive built-in modules

## Basic Data Types

### Numbers
Python supports several numeric types:
- **Integers (int)**: Whole numbers like 1, 42, -7
- **Floats (float)**: Decimal numbers like 3.14, -0.5
- **Complex numbers**: Like 3+4j

### Strings
Strings are sequences of characters enclosed in quotes:
- Single quotes: 'hello'
- Double quotes: "hello"
- Triple quotes for multi-line strings: '''hello world'''

### Lists
Lists are ordered, mutable collections:
```python
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
```

### Tuples
Tuples are ordered, immutable collections:
```python
coordinates = (10, 20)
```

### Dictionaries
Dictionaries store key-value pairs:
```python
person = {'name': 'John', 'age': 30, 'city': 'New York'}
```

## Control Flow

### If Statements
```python
if condition:
    # code block
elif another_condition:
    # code block
else:
    # code block
```

### For Loops
```python
for item in iterable:
    # code block
```

### While Loops
```python
while condition:
    # code block
```

## Functions
Functions are defined using the `def` keyword:
```python
def greet(name):
    return f"Hello, {name}!"
```

## Classes and Objects
Python supports object-oriented programming:
```python
class Dog:
    def __init__(self, name):
        self.name = name
    
    def bark(self):
        return "Woof!"
```

## Modules and Packages
- **Module**: A single Python file
- **Package**: A collection of modules in a directory with an `__init__.py` file

Import modules using:
```python
import math
from datetime import datetime
```

## Exception Handling
Handle errors gracefully:
```python
try:
    # risky code
except ValueError:
    # handle ValueError
except Exception as e:
    # handle other exceptions
finally:
    # always executed
```

## Best Practices
1. Follow PEP 8 style guide
2. Use meaningful variable names
3. Write docstrings for functions and classes
4. Keep functions small and focused
5. Handle exceptions appropriately
6. Use virtual environments for projects
