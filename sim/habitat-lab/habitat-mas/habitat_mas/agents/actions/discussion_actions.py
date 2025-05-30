from ..crab_core import action


@action
def eval_python_code(python_expression: str) -> str:
    """Execute the given python expression and return the result. You can use
    this function to do complex calculations.

    Args:
        python_code: The python expression to execute. Should strictly follow
            the python syntax.

    Returns:
        The eval result of the executed python code.
    """
    return str(eval(python_expression))


@action
def add(a: int, b: int) -> int:
    """Add two numbers and return the result.

    Args:
        a: The first number to add.
        b: The second number to add.

    Returns:
        The sum of the two numbers.
    """
    return a + b


@action
def subtract(a: int, b: int) -> int:
    """Subtract the second number from the first number and return the result.

    Args:
        a: The number to subtract from.
        b: The number to subtract.

    Returns:
        The difference of the two numbers.
    """
    return a - b


@action
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the result.

    Args:
        a: The first number to multiply.
        b: The second number to multiply.

    Returns:
        The product of the two numbers.
    """
    return a * b


@action
def divide(a: int, b: int) -> float:
    """Divide the first number by the second number and return the result.

    Args:
        a: The number to divide.
        b: The number to divide by.

    Returns:
        The quotient of the two numbers.
    """
    return a / b
