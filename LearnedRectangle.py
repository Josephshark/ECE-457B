"""
ECE 457B, Fall 2025
University of Waterloo
Mahesh Tripunitara, tripunit@uwaterloo.ca
"""

from generate_example import generate_example

"""
You are not allowed to import anything else.

generate_example() is a generator. If you invoke
next(generate_example()), it will give you an example.
An example is a pair (point, b) where b is True if point is a positive
example, and False if it is a negative example. The component point is
a list of length >= 1. You will discover the number of dimensions of
the target rectangle only after your first invocation to this
generator.

E.g., a return value from next(generate_example()) may be
([1,12,3], True). This means that the target rectangle is in
3-dimensions, and the point <1,12,3> is within that rectangle.
"""

class LearnedRectangle:
    def __init__(self):
        # Your code here to initialize the object.

    def learn(self, m):
        # Your code here to learn the target rectangle. You should
        # get the m examples by invoking next(generate_example()).
        # You can assume that m is an integer >= 1.

    def checkgoodness(self, n, k, epsilon):
        # Your code here for the following, whose intent is to check
        # the goodness of your learned rectangle.
        # Initialize a counter, and perform the following n times.
        # For k examples, check whether the proportion of those k
        # that are misclassified by your learned rectangle is > epsilon.
        # If yes, increase your counter by 1.
        # Return the value of your counter.
        #
        # E.g., suppose n = 2, k = 5 and epsilon = 0.2. This means
        # you will consider 2 sets of 5 examples each. Suppose for the
        # first # set of 5 examples, your learned rectangle has
        # misclassified 1 of those 5 examples. As # 1 <= 5 x 0.2, you
        # do not increase your counter. Suppose for the second set of
        # 5 examples, your learned rectangle misclassied 3 out of 5.
        # As 3 > 5 x 0.2, you will increase your counter by 1. And you
        # will return 1, i.e., the value of the counter, as your result.

        return 0 # replace with correct return value
