def generate_accuracy_steps(start: int = 50, size_steps: int = 10, stop: int = 90) -> list:
    number_steps = round((stop - start) / size_steps) + 1
    return [(start + size_steps * i)/100 for i in range(number_steps)]
