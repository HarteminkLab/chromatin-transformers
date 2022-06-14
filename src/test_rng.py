
import numpy as np


def main():
	rng = np.random.default_rng(123)
	my_list = np.arange(10)
	
	print("Using random number generator object:", rng.choice(my_list, size=2))

	np.random.seed(123)
	print("Using default random seed:", np.random.choice(my_list, size=2))


if __name__ == '__main__':
	main()
