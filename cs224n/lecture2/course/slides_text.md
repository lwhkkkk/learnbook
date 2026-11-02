# Stanford CS224N Lecture 2: Word Vectors and Language Models - Slides

## Slide 1: Natural Language Processing

with Deep Learning
CS224N/Ling284
DiyiYang
Lecture 2: Word Vectors, Word Senses, and Neural Classifiers

---

## Slide 2: Lecture Plan

2
Lecture 2: Word Vectors, Word Senses, and Neural Network Classifiers
1.Course organization (3 mins)
2.Finish looking at word vectors and word2vec (15 mins)
3.Can we capture the essence of word meaning more effectively by counting? (10m)
4.Evaluating word vectors (10 mins)
5.Word senses (8 mins)
6.Review of classification and how neural nets differ (14 mins)
7.Introducing neural networks (10 mins)
Key Goal: To be able to read word embeddings papers by the end of class

---

## Slide 3: 1. Course Organization

•Come to office hours/help sessions!
•They started yesterday
•Come to discuss final project ideas as well as the assignments
•Try to come early, often and off-cycle!
•TA office hours: 3-hour blocks Mon–Sat, with multiple TAs
•Just show up! Our friendly course staff will be on hand to assist you!
•https://web.stanford.edu/class/cs224n/office_hours.html
•Instructors’ office hours (in person by default):
•Diyi: Tuesdays 3-4pm
•Tatsu: Fridays 3-4pm
3

---

## Slide 4: 2. Review: Main idea of word2vec

5
•Start with random word vectors
•Iterate through each word position in the whole corpus
•Try to predict surrounding words using word vectors: 𝑃𝑜𝑐= !"#(%!"&#)∑$∈&!"#(%$"&#)
•Learning: Update vectors so they can predict actual surrounding words better
•Doing no more than this, this algorithm learns word vectors that capture
well word similarity and meaningful directions in a word space!
…crisesbankingintoturningproblems… as
𝑃𝑤!"#	|	𝑤!
𝑃𝑤!"$	|	𝑤!
𝑃𝑤!%#	|	𝑤!
𝑃𝑤!%$	|	𝑤!

---

## Slide 5: 2. Optimization: Gradient Descent

•We have a cost function 𝐽𝜃we want to minimize
•Gradient Descent is an algorithm to minimize 𝐽𝜃
•Idea: for current value of 𝜃, calculate gradient of 𝐽𝜃, then take small step in direction
of negative gradient. Repeat.
Note: Our
objectives
may not
be convex
like this L
But life turns
out to be
okay J
6

---

## Slide 6: •Update equation (in matrix notation):

•Update equation (for single parameter):
•Algorithm:
Gradient Descent
𝛼 = step size or learning rate
7

---

## Slide 7: Stochastic Gradient Descent

•Problem: 𝐽𝜃is a function of allwindows in the corpus (potentially billions!)
•So                 is very expensive to compute
•You would wait a very long time before making a single update!
•Verybad idea for pretty much all neural nets!
•Solution: Stochastic gradient descent (SGD)
•Repeatedly sample windows, and update after each one
•Algorithm:
8
Mini Batch
Gradient Descent

---

## Slide 8: Word2vec parameters         …            and computations

•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
U                           V 𝑈.𝑣)* softmax(𝑈.𝑣)*)
outside             center            dot product       probabilities
The model makes the same predictions at each position
9
We want a model that gives a reasonably high
probability estimate to all words that occur in the
context (at all often)
“Bag of words” model!

---

## Slide 9: Word2vec maximizes objective function by

putting similar words nearby in space
10

---

## Slide 10: Word2vec algorithm family (Mikolovet al. 2013): More details

Why two vectors? àEasier optimization. Average both at the end
•But can implement the algorithm with just one vector per word … and it helps a bit
Two model variants:
1.Skip-grams (SG)
Predict context (“outside”) words (position independent) given center word
2.Continuous Bag of Words (CBOW)
Predict center word from (bag of) context words
We presented: Skip-gram model
Loss functions for training:
1.Naïve softmax (simple but expensive loss function, when many output classes)
2.More optimized variants like hierarchical softmax
3.Negative sampling
So far, we explained naïve softmax
11

---

## Slide 11: The skip-gram model with negative sampling (HW2)

•The normalization term is computationally expensive (when many output classes):
•𝑃𝑜𝑐= !"#(%!"&#)∑$∈&!"#(%$"&#)
•Hence, in standard word2vec and HW2 you implement the skip-gram model with
negative sampling
•Main idea: train binary logistic regressions to differentiate a true pair (center word and
a word in its context window) versus several “noise” pairs (the center word paired with
a random word)
12
A big sum over words

---

## Slide 12: The skip-gram model with negative sampling (HW2)

13
•Introduced in: “Distributed Representations of Words and Phrases and their
Compositionality” (Mikolovet al. 2013)
•Overall objective function (they maximize):
•The logistic/sigmoid function:
(we’ll become good friends soon)
•We maximize the probability of two words
co-occurring in first log and minimize probability
of noise words in second part
sigmoid
rather than softmax

---

## Slide 13: The skip-gram model with negative sampling (HW2)

•Using notation consistent with this class and HW2:
𝐽+,-./0123,𝒖4,𝒗5,𝑈=−log𝜎𝒖4*𝒗5− :
6∈8/0123,9:+9:5,/
log𝜎−𝒖6*𝒗5
•We take knegative samples (using word probabilities)
•Maximize probability that real outside word appears;
minimize probability that random words appear around center word
•Sample with P(w)=U(w)3/4/Z, the unigram distribution U(w) raised to the 3/4 power
(We provide this function in the starter code).
•The power makes less frequent words be sampled more often
14
𝜎𝑥= 11+𝑒!"

---

## Slide 14: Stochastic gradients with negative sampling [aside]

•We iteratively take gradients at each window for SGD
•In each window, we only have at most 2m+ 1 words plus 2kmnegative
words with negative sampling, so ∇A𝐽B(𝜃)is very sparse!
15

---

## Slide 15: Stochastic gradients with with negative sampling [aside]

•We might only update the word vectors that actually appear!
•Solution: either you need sparse matrix update operations to
only update certain rowsof full embedding matrices Uand V,
or you need to keep around a hash for word vectors
•If you have millions of word vectors and do distributed
computing, it is important to not have to send gigantic
updates around!
[            ]|V|
d
16
Rows not columns
in actual DL
packages!

---

## Slide 16: 3. Why not capture co-occurrence counts directly?

17
There’s something weird about iterating through the whole corpus (perhaps many times);
why don’t we just accumulate all the statistics of what words appear near each other?!?
Building a co-occurrence matrix X
•2 options: windows vs. full document
•Window: Similar to word2vec, use window around each word àcaptures some
syntactic and semantic information (“word space”)
•Word-document co-occurrence matrix will give general topics (all sports terms will
have similar entries) leading to “Latent Semantic Analysis” (“document space”)

---

## Slide 17: Example: Window based co-occurrence matrix

18
•Window length 1 (more common: 5–10)
•Symmetric (irrelevant whether left or right context)
•Example corpus:
•I like deep learning
•I like NLP
•I enjoy flying
counts I like enjoydeep learningNLP flying .
I 0 2 1 0 0 0 0 0
like 2 0 0 1 0 1 0 0
enjoy 1 0 0 0 0 0 1 0
deep 0 1 0 0 1 0 0 0
learning0 0 0 1 0 0 0 1
NLP 0 1 0 0 0 0 0 1
flying 0 0 1 0 0 0 0 1
. 0 0 0 0 1 1 1 0

---

## Slide 18: Co-occurrence vectors

19
•Simple count co-occurrence vectors
•Vectors increase in size with vocabulary
•Very high dimensional: require a lot of storage (though sparse)
•Subsequent classification models have sparsity issues àModels are less robust
•Low-dimensional vectors
•Idea: store “most” of the important information in a fixed, small number of
dimensions: a dense vector
•Usually 25–1000 dimensions, similar to word2vec
•How to reduce the dimensionality?

---

## Slide 19: Classic Method: Dimensionality Reduction on X (HW1)

20
Singular Value Decomposition of co-occurrence matrix X
Factorizes Xinto UΣVT, where Uand Vare orthonormal (unit vectors and orthogonal)
Retain only k singular values, in order to generalize.
&𝑋 is the best rank k approximation to X , in terms of least squares.
Classic linear algebra result. Expensive to compute for large matrices.
kX

---

## Slide 20: Hacks to X (several used in Rohde et al. 2005 in COALS)

21
•Running an SVD on raw counts doesn’t work well!!!
•Scaling the counts in the cells can help a lot
•Problem: function words (the, he, has) are too frequent àsyntax has too much
impact. Some fixes:
• log the frequencies
• min(X, t), with t≈ 100
• Ignore the function words
•Ramped windows that count closer words more than further away words
•Use Pearson correlations instead of counts, then set negative values to 0
•Etc.

---

## Slide 21: Interesting semantic patterns emerge in the scaled vectorsRohde, Gonnerman, Plaut Modeling Word Meaning Using Lexical Co-Occurrence

DRIVE
LEARN
DOCTOR
CLEAN
DRIVER
STUDENT
TEACH
TEACHER
TREAT PRAY
PRIEST
MARRY
SWIM
BRIDE
JANITORSWIMMER
Figure 13: Multidimensional scaling for nouns and their associated verbs.
Table 10
The 10 nearest neighbors and their percent correlation similarities for a set of nouns, under the COALS-14K model.
gun point mind monopoly cardboard lipstick leningrad feet
1) 46.4 handgun 32.4 points 33.5 minds 39.9 monopolies 47.4 plastic 42.9 shimmery 24.0 moscow 59.5 inches
2) 41.1ﬁrearms 29.2 argument 24.9 consciousness 27.8 monopolistic 37.2 foam 40.8 eyeliner 22.7 sevastopol 57.7 foot
3) 41.0ﬁrearm 25.4 question 23.2 thoughts 26.5 corporations 36.7 plywood 38.8 clinique 22.7 petersburg 52.0 metres
4) 35.3 handguns 22.3 arguments 22.4 senses 25.0 government 35.6 paper 38.4 mascara 20.7 novosibirsk 45.7 legs
5) 35.0 guns 21.5 idea 22.2 subconscious 23.2 ownership 34.8 corrugated 37.2 revlon 20.3 russia 45.4 centimeters
6) 32.7 pistol 20.1 assertion 20.8 thinking 22.2 property 32.3 boxes 35.4 lipsticks 19.6 oblast 44.4 meters
7) 26.3 weapon 19.5 premise 20.6 perception 22.2 capitalism 31.3 wooden 35.3 gloss 19.5 minsk 40.2 inch
8) 24.4 riﬂes 19.3 moot 20.4 emotions 21.8 capitalist 31.0 glass 34.1 shimmer 19.2 stalingrad 38.4 shoulders
9) 24.2 shotgun 18.9 distinction 20.1 brain 21.6 authority 30.7 fabric 33.6 blush 19.1 ussr 37.8 knees
10) 23.6 weapons 18.7 statement 19.9 psyche 21.3 subsidies 30.5 aluminum 33.5 nars 19.0 soviet 36.9 toes
Table 11
The 10 nearest neighbors for a set of verbs, according to the COALS-14K model.
need buy play change send understand explain create
1) 50.4 want 53.5 buying 63.5 playing 56.9 changing 55.0 sending 56.3 comprehend 53.0 understand 58.2 creating
2) 50.2 needed 52.5 sell 55.5 played 55.3 changes 42.0 email 53.0 explain 46.3 describe 50.6 creates
3) 42.1 needing 49.1 bought 47.6 plays 48.9 changed 40.2 e-mail 49.5 understood 40.0 explaining 45.1 develop
4) 41.2 needs 41.8 purchase 37.2 players 32.2 adjust 39.8 unsubscribe 44.8 realize 39.8 comprehend 43.3 created
5) 41.1 can 40.3 purchased 35.4 player 30.2 affect 37.3 mail 40.9 grasp 39.7 explained 42.6 generate
6) 39.5 able 39.7 selling 33.8 game 29.5 modify 35.7 please 39.1 know 39.0 prove 37.8 build
7) 36.3 try 38.2 sells 32.3 games 28.3 different 33.3 subscribe 38.8 believe 38.2 clarify 36.4 maintain
8) 35.4 should 36.3 buys 29.0 listen 27.1 alter 33.1 receive 38.5 recognize 37.1 argue 36.4 produce
9) 35.3 do 34.0 sale 26.8 playable 25.6 shift 32.7 submit 38.0 misunderstand 37.0 refute 35.4 integrate
10) 34.7 necessary 31.5 cheap 25.0 beat 25.1 altering 31.5 address 37.9 understands 35.9 tell 35.2 implement
Table 12
The 10 nearest neighbors for a set of adjectives, according to the COALS-14K model.
high frightened red correct similar fast evil christian
1) 57.5 low 45.6 scared 53.7 blue 59.0 incorrect 44.9 similiar 43.1 faster 24.3 sinful 48.5 catholic
2) 51.9 higher 37.2 terri ﬁed 47.8 yellow 37.7 accurate 43.2 different 41.2 slow 23.4 wicked 48.1 protestant
3) 43.4 lower 33.7 confused 45.1 purple 37.5 proper 40.8 same 37.8 slower 23.2 vile 47.9 christians
4) 43.2 highest 33.3 frustrated 44.9 green 36.3 wrong 40.6 such 28.2 rapidly 22.5 demons 47.2 orthodox
5) 35.9 lowest 32.6 worried 43.2 white 34.1 precise 37.7 speci ﬁc 27.3 quicker 22.3 satan 47.1 religious
6) 31.5 increases 32.4 embarrassed 42.8 black 32.9 exact 35.6 identical 26.8 quick 22.3 god 46.4 christianity
7) 30.7 increase 32.3 angry 36.8 colored 30.7 erroneous 34.6 these 25.9 speeds 22.3 sinister 43.8 fundamentalist
8) 29.2 increasing 31.6 afraid 35.6 orange 30.6 valid 34.4 unusual 25.8 quickly 22.0 immoral 43.5 jewish
9) 28.7 increased 30.4 upset 33.5 grey 30.6 inaccurate 34.1 certain 25.5 speed 21.5 hateful 43.2 evangelical
10) 28.3 lowering 30.3 annoyed 32.4 reddish 29.8 acceptable 32.7 various 24.3 easy 21.3 sadistic 41.2 mormon
24
22
COALS model from
Rohde et al. ms., 2005. An Improved Model of Semantic Similarity Based on Lexical Co-Occurrence

---

## Slide 22: GloVe [Pennington, Socher, and Manning, EMNLP 2014]:

Encoding meaning components in vector differences
Q: How can we capture ratios of co-occurrence probabilities as
linear meaning components in a word vector space?

---

## Slide 23: A: Log-bilinear model:

with vector differences
GloVe [Pennington, Socher, and Manning, EMNLP 2014]:
Encoding meaning components in vector differences
Q: How can we capture ratios of co-occurrence probabilities as
linear meaning components in a word vector space?
• Fast training
• Scalable to huge corpora
Loss:

---

## Slide 24: 4. How to evaluate word vectors?

•Related to general evaluation in NLP: Intrinsic vs. extrinsic
•Intrinsic:
•Evaluation on a specific/intermediate subtask
•Fast to compute
•Helps to understand that system
•Not clear if really helpful unless correlation to real task is established
•Extrinsic:
•Evaluation on a real task
•Can take a long time to compute accuracy
•Unclear if the subsystem is the problem or its interaction or other subsystems
•If replacing exactly one subsystem with another improves accuracy àWinning!
25

---

## Slide 25: Intrinsic word vector evaluation

•Word Vector Analogies
•Evaluate word vectors by how well
their cosine distance after addition
captures intuitive semantic and
syntactic analogy questions
•Discarding the input words from the
search (!)
•Problem: What if the information is
there but not linear?
man:woman :: king:?
a:b :: c:?
king
man
woman
26

---

## Slide 26: GloVe Visualization

27

---

## Slide 27: Meaning similarity: Another intrinsic word vector evaluation

•Word vector distances and their correlation with human judgments
•Example dataset: WordSim353 http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/
28
Word 1Word 2Human (mean)
tiger cat 7.35
tiger tiger 10
book paper 7.46
computerinternet7.58
plane car 5.77
professordoctor 6.62
stock phone 1.62
stock CD 1.31
stock jaguar 0.92

---

## Slide 28: Correlation evaluation

•Word vector distances and their correlation with human judgments
the sumW+˜Was our word vectors. Doing so typ-
ically gives a small boost in performance, with the
biggest increase in the semantic analogy task.
We compare with the published results of a va-
riety of state-of-the-art models, as well as with
our own results produced using the word2vec
tool and with several baselines using SVDs. With
word2vec, we train the skip-gram (SG †) and
continuous bag-of-words (CBOW†) models on the
6 billion token corpus (Wikipedia 2014 + Giga-
word 5) with a vocabulary of the top 400,000 most
frequent words and a context window size of 10.
We used 10 negative samples, which we show in
Section 4.6 to be a good choice for this corpus.
For the SVD baselines, we generate a truncated
matrixXtruncwhich retains the information of how
frequently each word occurs with only the top
10,000 most frequent words. This step is typi-
cal of many matrix-factorization-based methods as
the extra columns can contribute a disproportion-
ate number of zero entries and the methods are
otherwise computationally expensive.
The singular vectors of this matrix constitute
the baseline “SVD”. We also evaluate two related
baselines: “SVD-S” in which we take the SVD ofpXtrunc, and “SVD-L” in which we take the SVD
of log(1+Xtrunc). Both methods help compress the
otherwise large range of values inX.7
4.3 Results
We present results on the word analogy task in Ta-
ble 2. The GloV e model performs signiﬁcantly
better than the other baselines, often with smaller
vector sizes and smaller corpora. Our results us-
ing theword2vectool are somewhat better than
most of the previously published results. This is
due to a number of factors, including our choice to
use negative sampling (which typically works bet-
ter than the hierarchical softmax), the number of
negative samples, and the choice of the corpus.
We demonstrate that the model can easily be
trained on a large 42 billion token corpus, with a
substantial corresponding performance boost. We
note that increasing the corpus size does not guar-
antee improved results for other models, as can be
seen by the decreased performance of the SVD-
7We also investigated several other weighting schemes for
transformingX; what we report here performed best. Many
weighting schemes like PPMI destroy the sparsity ofXand
therefore cannot feasibly be used with large vocabularies.
With smaller vocabularies, these information-theoretic trans-
formations do indeed work well on word similarity measures,
but they perform very poorly on the word analogy task.
Table 3: Spearman rank correlation on word simi-
larity tasks. All vectors are 300-dimensional. The
CBOW⇤vectors are from theword2vecwebsite
and differ in that they contain phrase vectors.
Model Size WS353 MC RG SCWS RW
SVD 6B 35.3 35.1 42.5 38.3 25.6
SVD-S 6B 56.5 71.5 71.0 53.6 34.7
SVD-L 6B 65.7 72.7 75.1 56.5 37.0
CBOW† 6B 57.2 65.6 68.2 57.0 32.5
SG† 6B 62.8 65.2 69.7 58.1 37.2
GloV e 6B 65.872.777.853.9 38.1
SVD-L 42B 74.0 76.4 74.1 58.3 39.9
GloV e 42B 75.983.682.959.647.8
CBOW⇤100B68.4 79.6 75.4 59.4 45.5
L model on this larger corpus. The fact that this
basic SVD model does not scale well to large cor-
pora lends further evidence to the necessity of the
type of weighting scheme proposed in our model.
Table 3 shows results on ﬁve different word
similarity datasets. A similarity score is obtained
from the word vectors by ﬁrst normalizing each
feature across the vocabulary and then calculat-
ing the cosine similarity. We compute Spearman’s
rank correlation coefﬁcient between this score and
the human judgments. CBOW⇤denotes the vec-
tors available on theword2vecwebsite that are
trained with word and phrase vectors on 100B
words of news data. GloV e outperforms it while
using a corpus less than half the size.
Table 4 shows results on the NER task with the
CRF-based model. The L-BFGS training termi-
nates when no improvement has been achieved on
the dev set for 25 iterations. Otherwise all conﬁg-
urations are identical to those used by Wang and
Manning (2013). The model labeled Discreteis
the baseline using a comprehensive set of discrete
features that comes with the standard distribution
of the Stanford NER model, but with no word vec-
tor features. In addition to the HPCA and SVD
models discussed previously, we also compare to
the models of Huang et al. (2012) (HSMN) and
Collobert and Weston (2008) (CW). We trained
the CBOW model using the word2vectool8.
The GloV e model outperforms all other methods
on all evaluation metrics, except for the CoNLL
test set, on which the HPCA method does slightly
better. We conclude that the GloV e vectors are
useful in downstream NLP tasks, as was ﬁrst
8We use the same parameters as above, except in this case
we found 5 negative samples to work slightly better than 10.
29

---

## Slide 29: Extrinsic word vector evaluation

•One example where good word vectors should help directly: named entity recognition: identifying
references to a person, organization or location: Chris Manning lives in Palo Alto.
Table 4: F1 score on NER task with 50d vectors.
Discreteis the baseline without word vectors. We
use publicly-available vectors for HPCA, HSMN,
and CW. See text for details.
Model Dev Test ACE MUC7
Discrete91.0 85.4 77.4 73.4
SVD 90.8 85.7 77.3 73.7
SVD-S 91.0 85.5 77.6 74.3
SVD-L 90.5 84.8 73.6 71.5
HPCA 92.688.7 81.7 80.7
HSMN 90.5 85.7 78.7 74.7
CW 92.2 87.4 81.7 80.2
CBOW 93.1 88.2 82.2 81.1
GloV e 93.288.3 82.9 82.2
shown for neural vectors in (Turian et al., 2010).
4.4 Model Analysis: V ector Length and
Context Size
In Fig. 2, we show the results of experiments that
vary vector length and context window. A context
window that extends to the left and right of a tar-
get word will be called symmetric, and one which
extends only to the left will be called asymmet-
ric. In (a), we observe diminishing returns for vec-
tors larger than about 200 dimensions. In (b) and
(c), we examine the effect of varying the window
size for symmetric and asymmetric context win-
dows. Performance is better on the syntactic sub-
task for small and asymmetric context windows,
which aligns with the intuition that syntactic infor-
mation is mostly drawn from the immediate con-
text and can depend strongly on word order. Se-
mantic information, on the other hand, is more fre-
quently non-local, and more of it is captured with
larger window sizes.
4.5 Model Analysis: Corpus Size
In Fig. 3, we show performance on the word anal-
ogy task for 300-dimensional vectors trained on
different corpora. On the syntactic subtask, there
is a monotonic increase in performance as the cor-
pus size increases. This is to be expected since
larger corpora typically produce better statistics.
Interestingly, the same trend is not true for the se-
mantic subtask, where the models trained on the
smaller Wikipedia corpora do better than those
trained on the larger Gigaword corpus. This is
likely due to the large number of city- and country-
based analogies in the analogy dataset and the fact
that Wikipedia has fairly comprehensive articles
for most such locations. Moreover, Wikipedia’s
50
55
60
65
70
75
80
85
OverallSyntacticSemantic
Wiki2010
1B tokens
Accuracy [%]
Wiki2014
1.6B tokens
Gigaword5
4.3B tokens
Gigaword5 +
Wiki2014
6B tokens
Common Crawl
42B tokens
Figure 3: Accuracy on the analogy task for 300-
dimensional vectors trained on different corpora.
entries are updated to assimilate new knowledge,
whereas Gigaword is a ﬁxed news repository with
outdated and possibly incorrect information.
4.6 Model Analysis: Run-time
The total run-time is split between populating X
and training the model. The former depends on
many factors, including window size, vocabulary
size, and corpus size. Though we did not do so,
this step could easily be parallelized across mul-
tiple machines (see, e.g., Lebret and Collobert
(2014) for some benchmarks). Using a single
thread of a dual 2.1GHz Intel Xeon E5-2658 ma-
chine, populatingXwith a 10 word symmetric
context window, a 400,000 word vocabulary, and
a 6 billion token corpus takes about 85 minutes.
GivenX, the time it takes to train the model de-
pends on the vector size and the number of itera-
tions. For 300-dimensional vectors with the above
settings (and using all 32 cores of the above ma-
chine), a single iteration takes 14 minutes. See
Fig. 4 for a plot of the learning curve.
4.7 Model Analysis: Comparison with
word2vec
A rigorous quantitative comparison of GloV e with
word2vecis complicated by the existence of
many parameters that have a strong effect on per-
formance. We control for the main sources of vari-
ation that we identiﬁed in Sections 4.4 and 4.5 by
setting the vector length, context window size, cor-
pus, and vocabulary size to the conﬁguration men-
tioned in the previous subsection.
The most important remaining variable to con-
trol for is training time. For GloV e, the rele-
vant parameter is the number of training iterations.
Forword2vec, the obvious choice would be the
number of training epochs. Unfortunately, the
code is currently designed for only a single epoch:
30

---

## Slide 30: 5. Word senses and word sense ambiguity

•Most words have lots of meanings!
•Especially common words
•Especially words that have existed for a long time
•Example: pike
•Does one vector capture all these meanings or do we have a mess?
31

---

## Slide 31: pike

•A sharp point or staff
•A type of elongated fish
•A railroad line or system
•A type of road
•The future (coming down the pike)
•A type of body position (as in diving)
•To kill or pierce with a pike
•To make one’s way (pike along)
•In Australian English, pike means to pull out from doing something: I reckon he could
have climbed that cliff, but he piked!
32

---

## Slide 32: Improving Word Representations Via Global Context And

Multiple Word Prototypes (Huang et al. 2012)
•Idea: Cluster word windows around words, retrain with each word assigned to multiple
different clusters bank1, bank2, etc.
33

---

## Slide 33: Linear Algebraic Structure of Word Senses, with

Applications to Polysemy      (Arora, …, Ma, …, TACL 2018)
•Different senses of a word reside in a linear superposition (weighted
sum) in standard word embeddings like word2vec
•𝑣pike=𝛼!𝑣pike#+𝛼"𝑣pike$+ 𝛼#𝑣pike%
•Where 𝛼!= $#$#%$$%$%, etc., for frequency f
•Surprising result:
•Because of ideas from sparse coding you can actually separate out
the senses (providing they are relatively common)!
34

---

## Slide 34: 6. Deep Learning Classification: Named Entity Recognition (NER)

•The task: findand classifynames in text, by labeling word tokens, for example:
Last night , Paris Hilton wowed in a sequin gown .
PER   PER
Samuel Quinn was arrested in the Hilton Hotel in Paris in April 1989 .
PER        PER                                         LOC     LOC        LOC      DATE DATE
•Possible uses:
•Tracking mentions of particular entities in documents
•For question answering, answers are usually named entities
•Relating sentiment analysis to the entity under discussion
•Often followed by Entity Linking/Canonicalization into a Knowledge Base such as Wikidata
35

---

## Slide 35: Simple NER: Window classification using binary logistic classifier

•Idea:classify each word in its context window of neighboring words
•Train logistic classifier on hand-labeled data to classify center word {yes/no} for each
class based on a concatenation of word vectors in a window
•Really, we usually use multi-class softmax, but we’re trying to keep it simple J
•Example: Classify “Paris” as +/–location in context of sentence with window length 2:
the     museums      in         Paris         are      amazing    to      see      .
Xwindow= [  xmuseums xin xParis xare xamazing]T
•Resulting vector xwindow= x ∈R5d
•To classify all words: run classifier for each class on the vector centered on each word
in the sentence
36

---

## Slide 36: NER: Binary classification for center word being location

40
•We do supervised training and want high score if it’s a location
𝐽B𝜃=𝜎𝑠= 11+𝑒./
x  = [  xmuseums xin         xParis            xare        xamazing ]
predicted model
probability of class
f = Some element-
wise non-linear
function, e.g.,
logistic, tanh, ReLU

---

## Slide 37: Remember: Stochastic Gradient Descent

Update equation:
i.e., for each parameter: 𝜃C+,D=𝜃C439−𝛼EFAEA’!()
In deep learning, 𝜃 includes the data representation (e.g., word vectors) too!
How can we compute ∇A𝐽(𝜃)?
1.By hand
2.Algorithmically: the backpropagation algorithm (next lecture!)
𝛼 = step size or learning rate
43

---

## Slide 38: 7. Neural computation

44

---

## Slide 39: A binary logistic regression unit is a bit similar to a neuron

hw,b(x)=f(wTx+b)
f(z)=11+e−z
w, b are the parameters of this neuron
i.e., this logistic regression model
b: We can have an “always on” bias
feature, which gives a class prior, or
separate it out, as a bias term
45
f = nonlinear activation function (e.g. sigmoid), w = weights, b = bias, h = hidden, x = inputs

---

## Slide 40: A neural network

= running several logistic regressions at the same time
If we feed a vector of inputs through a bunch of logistic regression functions, then we get
a vector of outputs …
But we don’t have to decide
ahead of time what variables
these logistic regressions are
trying to predict!
46

---

## Slide 41: A neural network

= running several logistic regressions at the same time
… which we can feed into another logistic regression function, giving composed functions
It is the loss function
that will direct what
the intermediate
hidden variables should
be, so as to do a good
job at predicting the
targets for the next
layer, etc.
47

---

## Slide 42: A neural network

= running several logistic regressions at the same time
Before we know it, we have a multilayer neural network….
48
This allows us to
re-represent and
compose our data
multiple times and to
learn a classifier that is
highly non-linear in
terms of the original
inputs
(but typically is linear in terms of
the pre-final layer representations)

---

## Slide 43: Matrix notation for a layer

We have
In matrix notation
Activation fis applied element-wise:
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
49

---

## Slide 44: Non-linearities (like for sigmoid): Why they’re needed

50
•Neural networks do function approximation,
e.g., regression or classification
•Without non-linearities, deep neural networks
can’t do anything more than a linear transform
•Extra layers could just be compiled down into a
single linear transform: W1W2x = Wx
•But, with more layers that include non-linearities,
they can approximate more complex functions!

---

## Slide 45: Slide 45



---

