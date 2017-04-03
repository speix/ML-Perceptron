# Machine Learning - Perceptron Supervised Learning
Implementation of the Perceptron Learning Algorithm (a.k.a PLA) for a linearly separable dataset.

### Usage

* Required libraries: numpy, matplotlib

```
python perceptron.py training-data.csv output.csv
```
### Results
With each PLA iteration, the program prints a new line to the output file, containing a comma-separated list of the weights and bias on each learning stage. Upon convergence, the program stops, prints the final values to the csv file and shows a figure with the given features and the dicision boundary (The Perceptron) that the PLA has computed.
![alt tag](http://www.supergramm.com/media/images/github/perceptron.png)
