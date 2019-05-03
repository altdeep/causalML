# Causality in Machine Learning

CS 7290 Special Topics in Data Science
Summer 2019
Prof. Robert Osazuwa Ness
Northeastern University, College of Computer and Information Science

[Syllabus and schedule](https://github.com/robertness/causalML/blob/master/syllabus.md)

[Class discussion forum](https://piazza.com/class/jv2j4bw56an62b)
  	
## Time and Place: Wednesday, 6pm, room: TBA

## Course Description

How can we model and create decision making agents that actually understand their environment? How will change to systems that serve ads, assign content to a feed, or direct a fleet of drivers actually impact bids/engagement/rides (despite what correlation tells us)?  How can we make sure our decision-making algorithms aren't amplifying the racism, sexism, and other societal-'isms' that bias the training data?

Causal inference answers these questions by enabling predictions about a system's behavior after an intervention, something mathematically impossible using deep learning or other advanced pattern recognition methods alone.  This is why data scientists with causal inference skills are increasingly sought after by the top tech companies.

This course introduces key ideas in causal inference and how to apply them to areas of machine learning where they have the most impact, such as reinforcement learning, agent modeling, and improving the performance of machine learning systems using past human-interaction data.  The focus is on how to put these ideas into practice with real world machine learning systems. 

One of the challenges of learning causal inference is that the domain is built on work by researchers from different fields including computer science, statistics, econometrics, epidemiology, and cognitive science.  These are different communities working on different sets of problems under different constraints.  Often, they use the same methods but under different names.  Just as often, they use different methods for similar problems and argue vehemently about which method is superior.  This course guides the learner past these challenges by using one causal inference framework (directed graphical models and structural causal models) to unite all the causal inference problems that typically pop up in machine learning.  When a causal inference problem in machine learning problem is commonly addressed by another framework, we will introduce it and contrast it with our unifying framework.

Topics include observational vs interventional data, traditional experiments vs quasi-experiments, counterfactual reasoning, prediction vs prediction under intervention, reasoning about confounders, asking causal questions of GANS and other implicit generative models, algorithmic bias, and both online and offline policy search and evaluation in bandits and reinforcement learning.

## Prerequisites

(DS5220 and DS5230) or (CS6140 and CS6220) or approval of the instructor

## Scribing

Each lecture will have a 2-3 assigned scribes who will be jointly responsible for producing notes in Markdown (we will make a template available). Each student will be assigned to two groups over the course of the semester. Notes will be due 1 week after the class takes place, and will be graded as part of the course work.

## Readings

While many great causal inference textbooks exist, none focus on the key problems in machine learning and AI that will be addressed in this class.  So instead we will focus heavily on assigned readings. Students are expected to read the assigned readings in advance of each lecture (stay tuned to this page for a reading list).

## Homework

The homework in this class will consist of 5 problem sets, which will combine mathematical derivations with programming exercises in Python. Submissions must be made via blackboard by 11.59pm on the due date.

## Project

The goal of the project is to gain experience in implementing, testing, and presenting one of the methods covered in the lectures. Students will collaborate in groups of 2-4. We will provide a list of suggested problems to choose from.

## Grading and Academic Guidelines

The final grade for this course will be weighted as follows:

    Homework: 40%
    Scribing: 20%
    Course Project: 40%

Refresh your knowledge of the university's [policy](http://www.northeastern.edu/osccr/academic-integrity-policy/) about academic integrity and plagiarism (this includes plagarizing code). There is **zero-tolerance** for cheating!

## Self-evaluation

Students will be asked to indicate the amount of time spent on each homework, as well as the project. The will also be able to indicate what they think went well, and what they think did not go well. There will also be an opportunity to provide feedback on the class after the midterm exam.

