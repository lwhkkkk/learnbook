# Stanford CS224N Lecture 3: Neural Networks and Backpropagation - Slides

## Slide 1: Natural Language Processing

with Deep Learning
CS224N/Ling284
Christopher Manning
Lecture 3: Neural net learning: Gradients by hand (matrix calculus)
and algorithmically (the backpropagation algorithm)

---

## Slide 2: 1. Introduction

2
Assignment 2 makes sure you really understand the math of neural networks … then we’ll
let the software do it! It also teaches us about dependency parsing.
We’ll go through all the math quickly today, but this is the one week of quarter to most
work through the readings!!!
This will be a tough week for some! à Make sure to get help if you need it:
Visit office hours! Read tutorial materials on the syllabus!
Thursday will be mainly linguistics! Some people find that tough too.
😉
PyTorch tutorial: 3:30pm Friday Apr 12 in Gates B01
A great chance to get an intro to PyTorch, a key deep learning package, used in Ass 2+!

---

## Slide 3: Where we ended last time: A neural network

= running several logistic regressions at the same time
Before we know it, we have a multilayer neural network….
3
This allows us to
re-represent and
compose our data
multiple times and to
learn a classifier that is
highly non-linear in
terms of the original
inputs
(but, typically, is linear in terms of
the pre-final layer representations)

---

## Slide 4: Matrix notation for a layer

We have
In matrix notation
Activation f is applied element-wise:
a1
a2
a3
a1=f(W11x1+W12x2+W13x3+b1)
a2=f(W21x1+W22x2+W23x3+b2)
etc.
z=Wx+b
a=f(z)
f([z1,z2,z3])=[f(z1),f(z2),f(z3)]
W12
b3
4

---

## Slide 5: NER: Binary classification for center word being location

•We do supervised training and want high score if it’s a location
𝐽!𝜃=𝜎𝑠= 11+𝑒"#
5
x  = [  xmuseums xin         xParis            xare        xamazing ]
predicted model
probability of class
f = Some element-
wise non-linear
function, e.g.,
logistic, tanh, ReLU
∈ R5d
Embedding of
1-hot words

---

## Slide 6: 7. Neural computation

6
Original McCulloch & Pitts
1943 threshold unit:
𝟏(𝑊𝑥>𝜃)
=	𝟏(𝑊𝑥−𝜃>0)
This function has no slope,
so, no gradient-based learning

---

## Slide 7: tanh is just a rescaled and shifted sigmoid (2×as steep, [−1,1]):

Logistic and tanh are still used (e.g., logistic to get a probability)
However, now, for deep networks, the first thing to try is ReLU: it
trains quickly and performs well due to good gradient backflow.
ReLU has a negative “dead zone” that recent proposals mitigate
GELU/Swish often used with Transformers (BERT, RoBERTa, etc.)
Non-linearities, old and new
logistic (“sigmoid”)                  tanh                       hard tanh                         ReLU
tanh(z)=2logistic(2z)−1
1
0
1
−1
Swish arXiv:1710.05941
swish𝑥=𝑥’logistic(𝑥)
ReLU𝑧=max(𝑧,0)
(Rectified Linear Unit) Leaky ReLU /
Parametric ReLU
0
0
0
GELU arXiv:1606.08415
GELU𝑥
=𝑥2𝑃𝑋≤𝑥,𝑋~𝑁(0,1)
≈𝑥2logistic(1.702𝑥)

---

## Slide 8: Non-linearities (i.e., “f ” on previous slide): Why they’re needed

8
•Neural networks do function approximation,
e.g., regression or classification
•Without non-linearities, deep neural networks
can’t do anything more than a linear transform
•Extra layers could just be compiled down into a
single linear transform: W1 W2 x = Wx
•But, with more layers that include non-linearities,
they can approximate any complex function!

---

## Slide 9: Remember: Stochastic Gradient Descent

Update equation:
i.e., for each parameter: 𝜃!"#$=𝜃!%&’−𝛼()*(*!"#$
In deep learning, 𝜃 includes the data representation (e.g., word vectors) too!
How can we compute ∇*𝐽(𝜃)?
1.By hand
2.Algorithmically: the backpropagation algorithm
𝛼 = step size or learning rate
9

---

## Slide 10: Lecture Plan

Lecture 4: Gradients by hand and algorithmically
1.Introduction (10 mins)
2.Matrix calculus (35 mins)
3.Backpropagation (35 mins)
Key Learning: The mathematics and practical implementation of how neural networks are
trained by backpropagation
10

---

## Slide 11: Computing Gradients by Hand

11
•Matrix calculus: Fully vectorized gradients
•“Multivariable calculus is just like single-variable calculus if you use matrices”
•Much faster and more useful than non-vectorized gradients
•But doing a non-vectorized gradient can be good for intuition; recall the first
lecture for an example
•Lecture notes and matrix calculus notes cover this material in more detail
•You might also review Math 51, which has an online textbook:
http://web.stanford.edu/class/math51/textbook.html

---

## Slide 12: 12



---

## Slide 13: Gradients

13
•Given a function with 1 output and 1 input
𝑓𝑥=𝑥+
•It’s gradient (slope) is its derivative
’,
’-=3𝑥.
“How much will the output change if we change the input a bit?”
At x = 1 it changes about 3 times as much: 1.013 = 1.03
At x = 4 it changes about 48 times as much: 4.013 = 64.48

---

## Slide 14: Gradients

•Given a function with 1 output and n inputs
•Its gradient is a vector of partial derivatives with
respect to each input
14

---

## Slide 15: Jacobian Matrix: Generalization of the Gradient

•Given a function with m outputs and n inputs
•It’s Jacobian is an m x n matrix of partial derivatives
15

---

## Slide 16: Chain Rule

•For composition of one-variable functions: multiply derivatives
•For multiple variables functions: multiply Jacobians
16

---

## Slide 17: Example Jacobian: Elementwise activation Function

17

---

## Slide 18: Example Jacobian: Elementwise activation Function

Function has n outputs and n inputs → n by n Jacobian
18

---

## Slide 19: Example Jacobian: Elementwise activation Function

19

---

## Slide 20: Example Jacobian: Elementwise activation Function

20

---

## Slide 21: Example Jacobian: Elementwise activation Function

21

---

## Slide 22: Other Jacobians

•Compute these at home for practice!
•Check your answers with the lecture notes
22

---

## Slide 23: Other Jacobians

•Compute these at home for practice!
•Check your answers with the lecture notes
23

---

## Slide 24: Other Jacobians

•Compute these at home for practice!
•Check your answers with the lecture notes
24
Fine print: This is the correct Jacobian.
Later we discuss the “shape convention”;
using it the answer would be h.

---

## Slide 25: Other Jacobians

•Compute these at home for practice!
•Check your answers with the lecture notes
25

---

## Slide 26: Back to our Neural Net!

x  = [  xmuseums xin         xParis            xare        xamazing ]
26

---

## Slide 27: Back to our Neural Net!

•Let’s find
•Really, we care about the gradient of the loss Jt but we will compute the
gradient of the score for simplicity (and because the logistic is in Ass 1!)
27
x  = [  xmuseums xin         xParis            xare        xamazing ]

---

## Slide 28: 1. Break up equations into simple pieces

28
Carefully define your variables and keep track of their dimensionality!

---

## Slide 29: 2. Apply the chain rule

29

---

## Slide 30: 2. Apply the chain rule

30

---

## Slide 31: 2. Apply the chain rule

31

---

## Slide 32: 2. Apply the chain rule

32

---

## Slide 33: 3. Write out the Jacobians

Useful Jacobians from previous slide
33

---

## Slide 34: 3. Write out the Jacobians

34
𝒖!
Useful Jacobians from previous slide

---

## Slide 35: 3. Write out the Jacobians

35
𝒖!
Useful Jacobians from previous slide

---

## Slide 36: 3. Write out the Jacobians

36
𝒖!
Useful Jacobians from previous slide

---

## Slide 37: 3. Write out the Jacobians

37
𝒖!
𝒖!
Useful Jacobians from previous slide
.
⊙	= Hadamard product =
element-wise multiplication
of 2 vectors to give vector

---

## Slide 38: Re-using Computation

•Suppose we now want to compute
•Using the chain rule again:
38

---

## Slide 39: Re-using Computation

•Suppose we now want to compute
•Using the chain rule again:
The same! Let’s avoid duplicated computation …
39

---

## Slide 40: Re-using Computation

•Suppose we now want to compute
•Using the chain rule again:
40
𝛿 is the upstream gradient (“error signal”)
𝒖!

---

## Slide 41: Derivative with respect to Matrix: Output shape

•What does             look like?
•1 output, nm inputs: 1 by nm Jacobian?
•Inconvenient to then do
41

---

## Slide 42: Derivative with respect to Matrix: Output shape

•What does             look like?
•1 output, nm inputs: 1 by nm Jacobian?
•Inconvenient to then do
•Instead, we leave pure math and use the shape convention:
the shape of the gradient is the shape of the parameters!
•So              is n by m:
42

---

## Slide 43: Derivative with respect to Matrix

•What is
•    is going to be in our answer
•The other term should be       because
•Answer is:
43
𝛿 is upstream gradient (“error signal”) at 𝑧
𝑥 is local input signal

---

## Slide 44: Why the Transposes?

44
•Hacky answer: this makes the dimensions work out!
•Useful trick for checking your work!
•Full explanation in the lecture notes
•Each input goes to each output – you want to get outer product

---

## Slide 45: Deriving local input gradient in backprop

•For "𝒛"𝑾 in our equation:
•Let’s consider the derivative of a single weight Wij
•Wij only contributes to zi
•For example: W23 is only
used to compute z2 not z1
45
x1            x2                 x3            +1
f(z1)=   h1            h2 =f(z2)
s  u2
W23
b2
𝜕𝑠
𝜕𝑾=𝜹𝜕𝒛𝜕𝑾=𝜹𝜕𝜕𝑾(𝑾𝒙+𝒃)
𝜕𝑧1
𝜕𝑊1!
=𝜕𝜕𝑊1!𝑾12𝒙+𝑏1
=((3%! ∑456’ 𝑊14𝑥4=𝑥!

---

## Slide 46: What shape should derivatives be?

•Similarly,                            is a row vector
•But shape convention says our gradient should be a column vector because b is
a column vector …
•Disagreement between Jacobian form (which makes the chain rule
easy) and the shape convention (which makes implementing SGD easy)
•We expect answers in the assignment to follow the shape convention
•But Jacobian form is useful for computing the answers
46

---

## Slide 47: What shape should derivatives be?

Two options for working through specific problems:
1.Use Jacobian form as much as possible, reshape to
follow the shape convention at the end:
•What we just did. But at the end transpose       to make the
derivative a column vector, resulting in
2.Always follow the shape convention
•Look at dimensions to figure out when to transpose and/or
reorder terms
•The error message 𝜹 that arrives at a hidden layer has the
same dimensionality as that hidden layer
47

---

## Slide 48: 3. Backpropagation

We’ve almost shown you backpropagation
It’s taking derivatives and using the (generalized, multivariate, or matrix)
chain rule
One more concept:
We re-use derivatives computed for higher layers in computing
derivatives for lower layers to minimize computation
48

---

## Slide 49: Computation Graphs and Backpropagation

 + 
•Software represents our neural
net equations as a graph
•Source nodes: inputs
•Interior nodes: operations
49

---

## Slide 50: Computation Graphs and Backpropagation

 + 
•Software represents our neural
net equations as a graph
•Source nodes: inputs
•Interior nodes: operations
•Edges pass along result of the
operation
50

---

## Slide 51: Computation Graphs and Backpropagation

 + 
•Software represents our neural
net equations as a graph
•Source nodes: inputs
•Interior nodes: operations
•Edges pass along result of the
operation
“Forward Propagation”
51

---

## Slide 52: Backpropagation

 + 
•Then go backwards along edges
•Pass along gradients
52

---

## Slide 53: Backpropagation: Single Node

•Node receives an “upstream gradient”
•Goal is to pass on the correct
“downstream gradient”
Upstream
gradient 53
Downstream
gradient

---

## Slide 54: Backpropagation: Single Node

Downstream
gradient
Upstream
gradient
•Each node has a local gradient
•The gradient of its output with
respect to its input
Local
gradient54

---

## Slide 55: Backpropagation: Single Node

Downstream
gradient
Upstream
gradient
•Each node has a local gradient
•The gradient of its output with
respect to its input
Local
gradient55
Chain
rule!

---

## Slide 56: Backpropagation: Single Node

Downstream
gradient
Upstream
gradient
•Each node has a local gradient
•The gradient of its output with
respect to its input
Local
gradient
•[downstream gradient] = [upstream gradient] x [local gradient]
56

---

## Slide 57: Backpropagation: Single Node

*
•What about nodes with multiple inputs?
57

---

## Slide 58: Backpropagation: Single Node

Downstream
gradients
Upstream
gradient
Local
gradients
*
•Multiple inputs → multiple local gradients
58

---

## Slide 59: An Example

59

---

## Slide 60: An Example

+
*
max
60
Forward prop steps

---

## Slide 61: An Example

+
*
max
61
Forward prop steps
6
3
2
1
2
2
0

---

## Slide 62: An Example

+
*
max
62
Forward prop steps
6
3
2
1
2
2
0
Local gradients

---

## Slide 63: An Example

+
*
max
63
Forward prop steps
6
3
2
1
2
2
0
Local gradients

---

## Slide 64: An Example

+
*
max
64
Forward prop steps
6
3
2
1
2
2
0
Local gradients

---

## Slide 65: An Example

+
*
max
65
Forward prop steps
6
3
2
1
2
2
0
Local gradients

---

## Slide 66: An Example

+
*
max
66
Forward prop steps
6
3
2
1
2
2
0
Local gradients
upstream * local = downstream
1
1*3 = 3
1*2 = 2

---

## Slide 67: An Example

+
*
max
67
Forward prop steps
6
3
2
1
2
2
0
Local gradients
upstream * local = downstream
1
3
2
3*1 = 3
3*0 = 0

---

## Slide 68: An Example

+
*
max
68
Forward prop steps
6
3
2
1
2
2
0
Local gradients
upstream * local = downstream
1
3
2
3
0
2*1 = 2
2*1 = 2

---

## Slide 69: An Example

+
*
max
69
Forward prop steps
6
3
2
1
2
2
0
Local gradients
1
3
2
3
0
2
2

---

## Slide 70: Gradients sum at outward branches

70
+

---

## Slide 71: Gradients sum at outward branches

71
+

---

## Slide 72: Node Intuitions

+
*
max
72
6
3
2
1
2
2
0
1
2
2
2
•+ “distributes” the upstream gradient to each summand

---

## Slide 73: Node Intuitions

+
*
max
73
6
3
2
1
2
2
0
1
33
0
•+ “distributes” the upstream gradient to each summand
•max “routes” the upstream gradient

---

## Slide 74: Node Intuitions

+
*
max
74
6
3
2
1
2
2
0
1
3
2
•+ “distributes” the upstream gradient
•max “routes” the upstream gradient
•* “switches” the upstream gradient

---

## Slide 75: Efficiency: compute all gradients at once

* + 
•Incorrect way of doing backprop:
•First compute
75

---

## Slide 76: Efficiency: compute all gradients at once

* + 
•Incorrect way of doing backprop:
•First compute
•Then independently compute
•Duplicated computation!
76

---

## Slide 77: Efficiency: compute all gradients at once

* + 
•Correct way:
•Compute all the gradients at once
•Analogous to using 𝜹 when we
computed gradients by hand
77

---

## Slide 78: 1.Fprop: visit nodes in topological sort order

-Compute value of node given predecessors
2.Bprop:
- initialize output gradient = 1
- visit nodes in reverse order:
Compute gradient wrt each node using
gradient wrt successors
Done correctly, big O() complexity of fprop and
bprop is the same
In general, our nets have regular layer-structure
and so we can use matrices and Jacobians…
Back-Prop in General Computation Graph
…
…
Inputs
= successors of
Single scalar output
78

---

## Slide 79: Automatic Differentiation

•The gradient computation can be
automatically inferred from the symbolic
expression of the fprop
•Each node type needs to know how to
compute its output and how to compute
the gradient wrt its inputs given the
gradient wrt its output
•Modern DL frameworks (Tensorflow,
PyTorch, etc.) do backpropagation for
you but mainly leave layer/node writer
to hand-calculate the local derivative
79

---

## Slide 80: Backprop Implementations

80

---

## Slide 81: Implementation: forward/backward API

81

---

## Slide 82: Implementation: forward/backward API

82

---

## Slide 83: Manual Gradient checking: Numeric Gradient

•For small h (≈ 1e-4),
•Easy to implement correctly
•But approximate and very slow:
•You have to recompute f for every parameter of our model
•Useful for checking your implementation
•In the old days, we hand-wrote everything, doing this everywhere was the key test
•Now much less needed; you can use it to check layers are correctly implemented
83

---

## Slide 84: Summary

84
We’ve mastered the core technology of neural nets!
🎉
🎉
🎉
•Backpropagation: recursively (and hence efficiently) apply the chain rule
along computation graph
• [downstream gradient] = [upstream gradient] x [local gradient]
•Forward pass: compute results of operations and save intermediate
values
•Backward pass: apply chain rule to compute gradients

---

## Slide 85: Why learn all these details about gradients?

85
•Modern deep learning frameworks compute gradients for you!
•Come to the PyTorch introduction this Friday!
•But why take a class on compilers or systems when they are implemented for you?
•Understanding what is going on under the hood is useful!
•Backpropagation doesn’t always work perfectly out of the box
•Understanding why is crucial for debugging and improving models
•See Karpathy article (in syllabus):
• https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
•Example in future lecture: exploding and vanishing gradients

---

