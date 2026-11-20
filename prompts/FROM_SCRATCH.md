Design a neural network from scratch for this classification task. An input has four real-valued coordinates x0, x1, x2, x3, each in [-1, 1], independently uniformly distributed. The correct class is 1 if x0 and x1 have the same nonzero sign; otherwise the correct class is 0, including if either is exactly zero. The other two coordinates are nuisance variables.

Use exactly this architecture with biases: 4 inputs, 8 first-layer hidden units, 8 second-layer hidden units, and 1 output. Both hidden layers use ReLU. The output is a scalar logit z. Predict class 1 if z > 0, otherwise class 0. Use output-by-input weight orientation:

h1 = ReLU(W1 @ x + b1)
h2 = ReLU(W2 @ h1 + b2)
z = (W3 @ h2 + b3)[0]

Supply all numerical weights and biases for this network, designed from scratch without training data or probes. No existing network, weights, or examples are provided. The matrices must be rectangular numeric arrays. Return only one JSON object, with exactly these keys and dimensions:
W1: 8 rows of 4 numbers; b1: 8 numbers;
W2: 8 rows of 8 numbers; b2: 8 numbers;
W3: 1 row of 8 numbers; b3: 1 number.
All numbers must be finite. Do not use ellipses, expressions, references, or extra keys. Return all 121 values explicitly. No markdown and no explanatory text.
