# Causal Modeling in Machine Learning

CS 7290 Special Topics in Data Science
Summer 2019
Prof. Robert Osazuwa Ness
Northeastern University, Khoury College of Computer Sciences

[Syllabus and schedule](https://github.com/robertness/causalML/blob/master/syllabus.md)

[Lecture notes](https://bookdown.org/connect/#/apps/2584/access)

[Class discussion forum](https://piazza.com/class/jv2j4bw56an62b)
  	
## Time and Place: Wednesday, 6pm, room: WVH 108

## Course Description

How can we model and create decision making agents that develop an understanding of their environment? How can we make sure bandit algorithms are led astray by some unexpected influence?  How can we measure the direct impact of a change to systems that serve ads, assign content to a feed, or direct a fleet of drivers impact bids/engagement/rides?  How can we make sure our predictive algorithms are not amplifying biases in the training data in a way that leads to disastrous decision-making?

Causal inference answers these questions by enabling predictions about a system's behavior after an intervention, something not possible even with the most advanced pattern recognition methods alone.  For this reason, the skillful application of causal inference is essential for the success of in-production machine learning at scale.

This course introduces key ideas in causal inference and how to apply them to areas of machine learning where they have the most impact, such as predicting under intervention, A/B testing, bandit algorithms, reinforcement learning, agent modeling, and improving the performance of machine learning systems using past human-interaction data.  The focus is on how to put these ideas into practice with real-world machine learning systems. 

This course targets data scientists and ML engineers familiar with probabilistic [generative modeling](https://en.wikipedia.org/wiki/Generative_model) in machine learning.  Learners familiar with how to use a tensor-based framework to build a Gaussian mixture model or a variational auto-encoder will find the material grafts directly onto this modeling intuition.  This course focuses on causal probabilistic modeling and structural causal models because they fit nicely into that generative model framework and its toolsets.  The course also covers elements of the [Neyman–Rubin causal model](https://en.wikipedia.org/wiki/Rubin_causal_model) that are commonplace in professional machine learning settings.

Learners who want to improve a multiarmed bandit or sequential decision process applied in a complex environment will also gain much from this course. 

Learners who seek to see causal inference language from the statistics, econometrics, epidemiology, and cognitive science literature united under a set of case studies relating to the engineering of large scale machine learning systems, have also come to the right place.

See the [syllabus](https://github.com/robertness/causalML/blob/master/syllabus.md) to learn more.

## Prerequisites

(DS5220 and DS5230) or (CS6140 and CS6220) or approval of the instructor

## Scribing

Each lecture will have a 2-3 assigned scribes who will be jointly responsible for producing notes in Markdown (we will make a template available). Each student will be assigned to two groups over the course of the semester. Notes will be due 1 week after the class takes place, and will be graded as part of the course work.

## Readings

This course does not require the purchase of textbooks.  However, it will rely heavily on the following two books:
* Pearl, Judea. Causality. Cambridge university press, 2009.
* Peters, Jonas, Dominik Janzing, and Bernhard Schölkopf. Elements of causal inference: foundations and learning algorithms. MIT Press, 2017.
While not necessary for the course, these books are worth buying just to have as a reference.

Another useful reference is Scutari and Denis's _Bayesian networks: with examples in R_, publish Chapman and Hall in 2014.  The book explains fundamental concepts in graphical modeling, Bayesian networks and causal Bayesian networks with examples using the bnlearn package in R.  The bnlearn package is the primary tool for causal discovery in the R community (with no real equivalent in Python), and by extension has become a popular general causal modeling tool as well.  That said, this course does not spend time on causal discovery algorithms and quickly moves beyond Bayes nets, so it may not be worth purchasing this book for this course.

These are useful for the fundamentals but will be relied on less as the class progresses.  We will focus heavily on papers at that point. 

Students are expected to read the assigned readings in advance of each lecture.


## Homework

The homework in this class will consist of 5 problem sets, which will combine mathematical derivations with programming exercises in Python. Submissions must be made via blackboard by 11.59pm on the due date.

## Project

The goal of the project is to gain experience in implementing, testing, and presenting one of the methods covered in the lectures. Students will collaborate in groups of 2-4. We will provide a list of project descriptions.  Students who want to pursue a unique project should speak to the instructor.  Unique projects done in collaboration with a company are encouraged.

## Grading and Academic Guidelines

The final grade for this course will be weighted as follows:

    Homework: 40%
    Scribing: 20%
    Course Project: 40%

Refresh your knowledge of the university's [policy](http://www.northeastern.edu/osccr/academic-integrity-policy/) about academic integrity and plagiarism (this includes plagarizing code). There is **zero-tolerance** for cheating!

## Self-evaluation

Students will be asked to indicate the amount of time spent on each homework, as well as the project. The will also be able to indicate what they think went well, and what they think did not go well.
