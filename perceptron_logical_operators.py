"""
Perceptrons as Logical Operators

Weights and bias for the AND, OR, NOT perceptrons
Our goal is to  set the weights (weight1, weight2) and bias to 
the correct values that calculate AND operation as shown above.
"""
import pandas as pd

# Set weight1, weight2, and bias
# 1. FOR AND: 
weight1 = 10.0
weight2 = 10.0
bias = -19.0

# 2. For OR:
#weight1 = 10.0, 20.0
#weight2 = 10.0, 20.0
#bias = -10.0, -19

# 3. For NOT:
#weight1 = 0.0
#weight2 = -11.0
#bias = 10.0

# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))

