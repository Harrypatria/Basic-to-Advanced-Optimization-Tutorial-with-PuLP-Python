# define the variables
var x1 binary;
var x2 binary;
var x3 binary;
var x4 binary;

# objective function
maximize proft: 12 * x1 + 8 * x2 + 7 * x3 + 6 * x4;

# constraints
subject to budget: 8 * x1 + 6 * x2 + 5 * x3 + 4 * x4 <= 15;
