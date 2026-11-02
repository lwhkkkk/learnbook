# Stanford CS224N Lecture 4: Syntactic Structure and Dependency Parsing - Slides

## Slide 1: Natural Language Processing
with Deep Le

Natural Language Processing
with Deep Learning 
CS224N/Ling284
Christopher Manning
Lecture 4: Dependency Parsing

---

## Slide 2: Lecture Plan
Syntactic Structure and Dep

Lecture Plan
Syntactic Structure and Dependency parsing
1.Syntactic Structure: Consistency and Dependency (30 mins)
2.Dependency Grammar and Treebanks (15 mins)
3.Transition-based dependency parsing (15 mins)
4.Neural dependency parsing (20 mins)
Key Learnings: Explicit linguistic structure and how a neural net can decide it
Reminders/comments: 
•In Assignment 2, you build a neural dependency parser using PyTorch! 
•Start installing and learning PyTorch (Ass 2 is quite scaffolded)
•Come to the PyTorch tutorial, Friday 3:30pm, Gates B01
•Final project discussions – come meet with us; focus of Tuesday class in week 4
•Chris office hours Mon 2:00-4:00pm on Calendly: https://calendly.com/manning/cs224n-office-hours 2

---

## Slide 3: 1. The linguistic structure of sentences

1. The linguistic structure of sentences – two views: Constituency 
= phrase structure grammar = context-free grammars (CFGs)
Phrase structure organizes words into nested constituents
Starting unit: words 
 the,   cat,   cuddly,   by,   door
Words combine into phrases
 the cuddly cat,       by the door
Phrases can combine into bigger phrases
 the cuddly cat by the door
 
3

---

## Slide 4: The linguistic structure of sentences –

The linguistic structure of sentences – two views: Constituency = 
phrase structure grammar = context-free grammars (CFGs)
Phrase structure organizes words into nested constituents.
                         the            cat
                         a                dog
                                 large                 in a crate
                                 barking            on the table
                                 cuddly              by the door
  large           barking
talk to
walked behind
6

---

## Slide 5: Two views of linguistic structure: Depen

Two views of linguistic structure: Dependency structure
•Dependency structure shows which words depend on (modify, attach to, or are 
arguments of) which other words.
Look  in  the  large  crate  in the kitchen by  the  door
7

---

## Slide 6: Why is sentence structure needed for com

Why is sentence structure needed for communication?
Humans communicate complex ideas by composing words together 
into bigger units to convey complex meanings
Human listeners need to work out what modifies [attaches to] what
A model needs to understand sentence structure in order to be able 
to interpret language correctly
9

---

## Slide 7: Prepositional phrase attachment ambiguit

Prepositional phrase attachment ambiguity
10

---

## Slide 8: Prepositional phrase attachment ambiguit

Prepositional phrase attachment ambiguity
Scientists count whales from space
Scientists count whales from space
✓
❌
11

---

## Slide 9: PP attachment ambiguities multiply
•A ke

PP attachment ambiguities multiply
•A key parsing decision is how we ‘attach’ various constituents
•PPs, adverbial or participial phrases, infinitives, coordinations, 
etc.
• Catalan numbers: Cn = (2n)!/[(n+1)!n!]
• An exponentially growing series, which arises in many tree-like contexts:
•E.g., the number of possible triangulations of a polygon with n+2 sides
•Turns up in triangulation of probabilistic graphical models (CS228)….
12

---

## Slide 10: Coordination scope ambiguity
Shuttle vet

Coordination scope ambiguity
Shuttle veteran and longtime NASA executive Fred Gregory appointed to board
Shuttle veteran and longtime NASA executive Fred Gregory appointed to board
14

---

## Slide 11: Coordination scope ambiguity
15

Coordination scope ambiguity
15

---

## Slide 12: Adjectival/Adverbial Modifier Ambiguity

Adjectival/Adverbial Modifier Ambiguity
16

---

## Slide 13: Verb Phrase (VP) attachment ambiguity
17

Verb Phrase (VP) attachment ambiguity
17

---

## Slide 14: Dependency paths help extract semantic i

Dependency paths help extract semantic interpretation – 
simple practical example: extracting protein-protein interaction
KaiC çnsubj  interacts  nmod:with è SasA
KaiC çnsubj  interacts nmod:with è SasA  conj:andè KaiA
KaiC çnsubj  interacts nmod:with è SasA  conj:andè KaiB
 [Erkan et al. EMNLP 07, Fundel et al. 2007, etc.]
demonstrated
results
KaiC
interacts
rythmically
nsubj
The
mark
det
ccomp
thatnsubj 
KaiBKaiA
SasA
conj:and
conj:and
advmod
nmod:with
with and
cc
case
18

---

## Slide 15: 2. Dependency Grammar and Dependency Str

2. Dependency Grammar and Dependency Structure
Dependency syntax postulates that syntactic structure consists of relations between 
lexical items, normally binary asymmetric relations (“arrows”) called dependencies
submitted
Bills were
Senatorby
immigration
Brownback
andon
ports
Republican
of
Kansas
19

---

## Slide 16: Dependency Grammar and Dependency Struct

Dependency Grammar and Dependency Structure
Dependency syntax postulates that syntactic structure consists of relations between 
lexical items, normally binary asymmetric relations (“arrows”) called dependencies
The arrows are 
commonly typed 
with the name of 
grammatical 
relations (subject, 
prepositional object, 
apposition, etc.)
submitted
Bills were
Senatorby
nsubj:pass aux obl
case
immigration
conj
Brownback
cc
andon
case
nmod
ports flat
Republican
of
case
nmod
Kansas
appos
20

---

## Slide 17: Dependency Grammar and Dependency Struct

Dependency Grammar and Dependency Structure
Dependency syntax postulates that syntactic structure consists of relations between 
lexical items, normally binary asymmetric relations (“arrows”) called dependencies
An arrow connects a head 
(governor, superior, 
regent) with a dependent 
(modifier, inferior, 
subordinate)
Usually, dependencies 
form a tree (a connected, 
acyclic, single-root graph)
submitted
Bills were
Senatorby
nsubj:pass aux obl
case
immigration
conj
Brownback
cc
andon
case
nmod
ports flat
Republican
of
case
nmod
Kansas
appos
21

---

## Slide 18: Dependency Grammar/Parsing History
•The

Dependency Grammar/Parsing History
•The idea of dependency structure goes back a long way
•To Pāṇini’s grammar (composed c. 5th century BCE)
•Basic approach of 1st millennium Arabic grammarians
•Constituency/context-free grammar is a new-fangled invention
•20th century invention (R.S. Wells, 1947; then Chomsky 1953, etc.)
•Modern dependency work is often sourced to Lucien Tesnière (1959)
•Was dominant approach in “East” in 20th Century (Russia, China, …)
•Good for free-er word order, inflected languages like Russian (or Latin!)
•Used in some of the earliest parsers in NLP, even in the US:
•David Hays, one of the founders of U.S. computational linguistics, built early (first?) 
dependency parser (Hays 1962) and published on dependency grammar in Language
22

---

## Slide 19: Pāṇini’s grammar (composed c. 5th centur

Pāṇini’s grammar (composed c. 5th century BCE)
Gallery:http://wellcomeimages.org/indexplus/image/L0032691.html
CC BY 4.0 File:Birch bark MS from Kashmir of the Rupavatra Wellcome L0032691.jpg
But this comes from much later – originally the grammar was oral
23

---

## Slide 20: ROOT Discussion of the outstanding issue

ROOT Discussion of the outstanding issues was completed  .
•Some people draw the arrows one way; some the other way! 
•Tesnière had them point from head to dependent – we follow that convention
•We usually add a fake ROOT so every word is a dependent of precisely 1 other node
Dependency Grammar and Dependency Structure
24

---

## Slide 21: The rise of annotated data & Universal D

The rise of annotated data & Universal Dependencies treebanks
Brown corpus (1967; PoS tagged 1979); Lancaster-IBM Treebank (starting late 1980s); 
Marcus et al. 1993, The Penn Treebank, Computational Linguistics;
Universal Dependencies: http://universaldependencies.org/
25

---

## Slide 22: The rise of annotated data
Starting off,

The rise of annotated data
Starting off, building a treebank seems a lot slower and less useful than writing a grammar 
(by hand)
But a treebank gives us many things
•Reusability of the labor
•Many parsers, part-of-speech taggers, etc. can be built on it
•Valuable resource for linguistics
•Broad coverage, not just a few intuitions
•Frequencies and distributional information
•A way to evaluate NLP systems
26

---

## Slide 23: What are the straightforward sources of

What are the straightforward sources of information for dependency parsing?
1.Bilexical affinities    The dependency [discussion à issues] is plausible
2.Dependency distance   Most (but not all) dependencies are between nearby words
3.Intervening material     Dependencies rarely span intervening verbs or punctuation
4.Valency of heads   How many dependents on which side are usual for a head?
ROOT Discussion of the outstanding issues was completed  .
Dependency Conditioning Preferences
27

---

## Slide 24: Dependency Parsing
•A sentence is parsed

Dependency Parsing
•A sentence is parsed by choosing for each word what other word (including ROOT) it is 
a dependent of
•Usually some constraints:
•Only one word is a dependent of ROOT
•Don’t want cycles A → B, B → A
•This makes the dependencies a tree
•Final issue is whether arrows can cross (be non-projective) or not
I give a on neuraltalk tomorrowROOT ’ll networks
28

---

## Slide 25: •Definition of a projective parse: There

•Definition of a projective parse: There are no crossing dependency arcs when the 
words are laid out in their linear order, with all arcs above the words
•Dependencies corresponding to a CFG tree must be projective
•I.e., by forming dependencies by taking 1 child of each category as head
•Most syntactic structure is projective like this, but dependency theory normally does 
allow non-projective structures to account for displaced constituents
•You can’t easily get the semantics of certain constructions right without these 
nonprojective dependencies
Who  did  Bill  buy  the  coffee  from  yesterday  ?
Projectivity
29

---

## Slide 26: 3. Methods of Dependency Parsing
1.Dynam

3. Methods of Dependency Parsing
1.Dynamic programming
Eisner (1996) gives a clever algorithm with complexity O(n3), by producing parse items 
with heads at the ends rather than in the middle
2.Graph algorithms
You create a Minimum Spanning Tree for a sentence
McDonald et al.’s (2005) O(n2) MSTParser scores dependencies independently using an 
ML classifier (he uses MIRA, for online learning, but it can be something else)
Neural graph-based parser: Dozat and Manning (2017) et seq. – very successful!
3.Constraint Satisfaction 
Edges are eliminated that don’t satisfy hard constraints. Karlsson (1990), etc.
4.“Transition-based parsing” or “deterministic dependency parsing”
Greedy choice of attachments guided by good machine learning classifiers
E.g., MaltParser (Nivre et al. 2008). Has proven highly effective. And fast.
30

---

## Slide 27: Greedy transition-based parsing [Nivre 2

Greedy transition-based parsing [Nivre 2003]
•A simple form of a greedy discriminative dependency parser
•The parser does a sequence of bottom-up actions
•Roughly like “shift” or “reduce” in a shift-reduce parser – CS143, anyone?? – but the 
“reduce” actions are specialized to create dependencies with head on left or right
•The parser has:
•a stack σ, written with top to the right
•which starts with the ROOT symbol
•a buffer β, written with top to the left
•which starts with the input sentence
•a set of dependency arcs A
•which starts off empty
•a set of actions
31

---

## Slide 28: Basic transition-based dependency parser

Basic transition-based dependency parser
Start: σ = [ROOT], β = w1, …, wn , A = ∅ 
1.Shift              σ, wi|β, A è σ|wi, β, A
2.Left-Arcr      σ|wi|wj, β, A è σ|wj, β, A∪{r(wj,wi)} 
3.Right-Arcr    σ|wi|wj, β, A è σ|wi, β, A∪{r(wi,wj)}
Finish: σ = [w], β = ∅
 
32

---

## Slide 29: Arc-standard transition-based parser
(th

Arc-standard transition-based parser
(there are other transition schemes …)
Analysis of “I ate fish”
atefish[root]
Start
I
[root]
Shift
I atefish
ate[root] fish
Shift
I
Start: σ = [ROOT], β = w1, …, wn , A = ∅ 
1. Shift              σ, wi|β, A è σ|wi, β, A
2. Left-Arcr      σ|wi|wj, β, A è 
                                 σ|wj, β, A∪{r(wj,wi)} 
3. Right-Arcr    σ|wi|wj, β, A è 
                                 σ|wi, β, A∪{r(wi,wj)}
Finish: σ = [w], β = ∅ 
33

---

## Slide 30: Arc-standard transition-based parser
Ana

Arc-standard transition-based parser
Analysis of “I ate fish”
ate[root] ate[root]
Left Arc
I
A +=
nsubj(ate → I)
atefish[root] atefish[root]
Shift
ate[root] [root]
Right Arc A +=
obj(ate → fish)fish ate
ate[root] [root]
Right Arc A +=
root([root] → ate)
Finish34
A = { nsubj(ate → I),
          obj(ate → fish)
          root([root] → ate) } 
Nota bene:
In this example 
I’ve at each step 
made the 
“correct” next 
transition.
But a parser has 
to work this out 
– by exploring or 
inferring!

---

## Slide 31: MaltParser [Nivre and Hall 2005]
•We hav

MaltParser [Nivre and Hall 2005]
•We have left to explain how we choose the next action 
🤷
•Answer: Stand back, I know machine learning!
•Each action is predicted by a discriminative classifier (e.g., softmax classifier) over each 
legal move
•Max of 3 untyped choices (max of |R| × 2 + 1 when typed)
•Features: top of stack word, POS; first in buffer word, POS; etc.
•There is NO search (in the simplest form)
•But you can profitably do a beam search if you wish (slower but better): 
•You keep k good parse prefixes at each time step
•The model’s accuracy is a bit below the state of the art in dependency parsing, but
•It provides very fast linear time parsing, with high accuracy – great for parsing the web
35

---

## Slide 32: Conventional Feature Representation
Feat

Conventional Feature Representation
Feature templates: usually a combination of 1–3 
elements from the configuration
Indicator features
00010010 0010binary, sparsedim =106 –107 …
36

---

## Slide 33: Evaluation of Dependency Parsing: (label

Evaluation of Dependency Parsing: (labeled) dependency accuracy
                  ROOT   She  saw   the   video   lecture 
                       0         1      2       3         4            5
Gold
1  2  She   nsubj
2  0  saw   root 
3  5  the   det
4  5  video  nn
5  2    lecture obj
Parsed
1  2  She   nsubj
2  0  saw   root 
3  4  the   det
4  5  video  nsubj
5  2    lecture ccomp 
Acc  =   # correct deps
     # of deps
UAS =  4 / 5  =  80%
LAS  =  2 / 5  =  40%
37

---

## Slide 34: Handling non-projectivity
•The arc-stand

Handling non-projectivity
•The arc-standard algorithm we just presented only builds projective dependency trees
•Possible directions to head:
1.Just declare defeat on nonprojective arcs 
🤷
2.Use dependency formalism which only has projective representations
•A CFG only allows projective structures; you promote head of projectivity violations
3.Use a postprocessor to a projective dependency parsing algorithm to identify and 
resolve nonprojective links
4.Add extra transitions that can model at least most non-projective structures (e.g., 
add an extra SWAP transition will allow any non-projectivity, cf. bubble sort)
5.Move to a parsing mechanism that does not use or require any constraints on 
projectivity (e.g., the graph-based MSTParser or Dozat and Manning (2017))
38

---

## Slide 35: 4. Why do we gain from a neural dependen

4. Why do we gain from a neural dependency parser?
Indicator Features Revisited
•Problem #1:  sparse
•Problem #2:  incomplete
•Problem #3: expensive to compute
More than 95% of parsing time is 
consumed by feature computation
NeuralApproach:
learn a dense and compact feature representation
0.1densedim = ~1000 0.9-0.20.3 -0.1-0.5…
39
Categorical features are:

---

## Slide 36: A neural dependency parser [Chen and Man

A neural dependency parser [Chen and Manning 2014]
•Results on English parsing to Stanford Dependencies:
•Unlabeled attachment score (UAS) = head
•Labeled attachment score (LAS) = head and label
Parser UAS LAS sent. / s
MaltParser 89.8 87.2 469
MSTParser 91.4 88.1 10
TurboParser 92.3 89.6 8
C & M 2014 92.0 89.7 654
40

---

## Slide 37: First win: Distributed Representations
•

First win: Distributed Representations
•We represent each word as a d-dimensional dense vector 
(i.e., word embedding)
•Similar words are expectedto have close vectors.
•Meanwhile, part-of-speech tags(POS) and dependency 
labelsare also represented as d-dimensional vectors.
•The smaller discrete sets also exhibit many semantical 
similarities.
come
go
werewas
is
good
NNS (plural noun) should be close toNN (singular noun).
nummod(numerical modifier) should be close toamod (adjective modifier).
41

---

## Slide 38: Extracting Tokens & vector representatio

Extracting Tokens & vector representations from configuration
•We extract a set of tokens based on the stack / buffer positions:
s1
s2
b1
lc(s1)
rc(s1)
lc(s2)
rc(s2)
good
has
control
∅
∅
He
∅
JJ
VBZ
NN
∅
∅
PRP
∅
∅
∅
∅
∅
∅
nsubj
∅
+ +
word POS dep.
}A concatenation of the vector representation of all these is the neural representation of 
a configuration
42

---

## Slide 39: Second win: Deep Learning classifiers ar

Second win: Deep Learning classifiers are non-linear classifiers
•Traditional ML classifiers (including Naïve Bayes, SVMs, logistic regression and softmax 
classifier) are not very powerful classifiers: they only give linear decision boundaries
•But neural networks can use multiple layers to learn much more complex nonlinear 
decision boundaries
     
43


---

## Slide 40: Neural Dependency Parser Model Architect

Neural Dependency Parser Model Architecture
(A simple feed-forward neural network multi-class classifier)
Input layer x
lookup + concat
Hidden layer h
h = ReLU(Wx + b1)
Output layer y
y = softmax(Uh + b2)
Softmax probabilities
44
{ Shift , Left-Arcr , Right-Arcr }
Log loss (cross-entropy error)will be back-
propagated to the embeddings
The hidden layer re-represents the input —
it moves inputs around in an intermediate 
layer vector space—so it can be easily 
classified with a (linear) softmax
Wins:
Distributed representations!
Non-linear classifier!

---

## Slide 41: Dependency parsing for sentence structur

Dependency parsing for sentence structure
45
Chen & Manning (2014) showed that neural networks can accurately 
determine the structure of sentences, supporting meaning interpretation
This paper was the first simple and successful neural dependency parser
The dense representations (and non-linear classifier) let it outperform other 
greedy parsers in both accuracy and speed


---

## Slide 42: Further developments in transition-based

Further developments in transition-based neural dependency parsing
46
This work was further developed and improved by others, including in particular at Google
•Bigger, deeper networks with better tuned hyperparameters
•Beam search
•Global, conditional random field (CRF)-style inference over the decision sequence
Leading to SyntaxNet and the Parsey McParseFace model (2016):
“The World’s Most Accurate Parser”
https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html 
Method UAS LAS (PTB WSJ SD 3.3)
Chen & Manning 2014 92.0 89.7
Weiss et al. 2015 93.99 92.05
Andor et al. 2016 94.61 92.79


---

## Slide 43: Graph-based dependency parsers
47
•Compu

Graph-based dependency parsers
47
•Compute a score for every possible dependency for each word
•Doing this well requires good “contextual” representations of each word token, 
which we will develop in coming lectures 
  ROOT          The            big            cat          sat
0.5
0.3
0.8
2.0
e.g., picking the head for “big”

---

## Slide 44: Graph-based dependency parsers
48
•Compu

Graph-based dependency parsers
48
•Compute a score for every possible dependency (choice of head) for each word
•Doing this well requires more than just knowing the two words 
•We need good “contextual” representations of each word token, which we will 
develop in the coming lectures 
•Repeat the same process for each other word; find the best parse (MST algorithm)
  ROOT          The            big            cat          sat
0.5
0.3
0.8
2.0
e.g., picking the head for “big”

---

## Slide 45: A Neural graph-based dependency parser
[

A Neural graph-based dependency parser
[Dozat and Manning 2017; Dozat, Qi, and Manning 2017]
49
•This paper revived interest in graph-based dependency parsing in a neural world
•Designed a biaffine scoring model for neural dependency parsing
•Also crucially uses a neural sequence model, something we discuss next week
•Really great results!
•But slower than the simple neural transition-based parsers
•There are n2 possible dependencies in a sentence of length n
Method UAS LAS (PTB WSJ SD 3.3)
Chen & Manning 2014 92.0 89.7
Weiss et al. 2015 93.99 92.05
Andor et al. 2016 94.61 92.79
Dozat & Manning 201795.74 94.08


---

