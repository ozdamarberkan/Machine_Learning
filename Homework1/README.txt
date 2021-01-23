To run code:

- Open Command Prompt as administrator.
- Go to the current directory where the main.py exist.
- Type python main.py to execute the code.

Note: The code runs for 10 mins.

How to read the outputs:

The outputs are stored in .csv files as required. The list of csv files are:

- feature_set.csv 		: The M matrix discussed in the question
- test_accuracy.csv 		: The accuracy for naive bayes with laplace smoothing a = 0.
- test_accuracy_laplace 	: The accuracy for naive bayes with laplace smoothing a = 1.
- forward_selection.csv 	: The indeces which the forward selection algorithm picks.
- frequency_selection.csv 	: The accuracies found with frequency selection algorithm.
