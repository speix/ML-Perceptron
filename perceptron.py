import argparse
import numpy
from math import copysign
import matplotlib.pyplot as plt


def train(data):

    bias = 0
    rate = 0.1
    weights = numpy.linspace(0, 0, 2)

    while True:

        errors = 0

        for x1, x2, label in data:

            y = classify([x1, x2], weights, bias)

            if y != label:

                d = label - y

                weights = [weight + (rate * d * feature)
                           for weight, feature in zip(weights, [x1, x2])]

                bias += rate * d

                errors += 1

        if errors == 0:
            plot(data, weights, bias)
            break

    return lambda x: classify([x1, x2], weights, bias)


def classify(sample, weights, bias):

    summary = sum(
            weight * feature for weight, feature in zip(weights, sample)
        ) + bias

    return copysign(1, summary)


def plot(data, weights, bias):

    x = numpy.linspace(0, 15)
    y = (bias + weights[0] * x) / -weights[1]
    plt.plot(x, y)

    for x1, x2, l in data:
        plt.plot(x1, x2, 'ro' if l == 1 else 'bo')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')

    args = parser.parse_args()

    data = numpy.genfromtxt(args.input, delimiter=',')

    train(data)

if __name__ == '__main__':
    main()
