import numpy as np

def generate_random_embedding(dimensions=512):
    # Generate a random vector with a specified number of dimensions
    embedding = np.random.rand(dimensions)
    return embedding

# Example usage
example_embedding = generate_random_embedding()
print("Random Embedding (Example):", example_embedding)
