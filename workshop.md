# **Workshop: Causal Modeling in Machine Learning**  
**Instructor:** Robert Osazuwa Ness  
*Author of* [***Causal AI***](https://www.robertosazuwaness.com/causal-ai-book/)

## **Overview**

This full-day workshop introduces participants to the foundations and modern applications of **causal modeling** in machine learning. We focus on practical, model-based reasoning using interventions, causal graphs, and counterfactuals — culminating in how these tools integrate with contemporary machine learning, including deep generative and probabilistic models. This workshop is a companion to [Probabilistic Machine Learning Workshop](https://github.com/altdeep/probmodeler/blob/main/README.md).

The workshop is designed for practitioners and researchers who want to build ML systems that reason robustly about *cause and effect*, go beyond correlations, and support decision-making in complex real-world environments.

---

## **Learning Objectives**

By the end of this workshop, participants will be able to:

- Construct causal graphs and reason about dependencies, interventions, and confounding.
- Apply do-calculus to determine identifiability of causal effects.
- Estimate causal effects using regression adjustment, propensity methods, and modern ML estimators.
- Perform counterfactual reasoning using structural causal models and algorithmic counterfactuals.
- Integrate causal reasoning with A/B testing, bandits, reinforcement learning, and sequential decision-making.
- Understand how deep latent variable models and probabilistic programming support counterfactual inference.

---

## **Target Audience**

This workshop is well suited for:

- ML researchers and data scientists  
- Experimentation platform teams  
- Applied scientists in tech, healthcare, finance, social science, or policy  
- PhD students and academics studying causal inference or causal ML  
- Engineers integrating causal reasoning into products, decision systems, or simulations

Participants should be comfortable with probability, random variables, and basic machine learning.

---

## **Schedule (Adjustable)**

### **8:00–8:30 — Introduction & Motivation**
- Why causality is essential in modern ML  
- Examples where predictive accuracy fails but causal reasoning succeeds  
- Causal ML in industry: experimentation, personalization, simulation, safety-critical systems

---

### **8:30–9:45 — Causal Graphs & Structural Models**
- Directed acyclic graphs (DAGs)  
- Causal vs. statistical relationships  
- Structural causal models (SCMs)  
- Conditional independence, d-separation, and graphical reasoning  
- Case study: confounding and backdoor criteria

---

### **9:45–10:00 — Break**

---

### **10:00–11:30 — Interventions & Causal Inference**
- The do-operator and intervention semantics  
- Identification of causal effects  
- Adjustment sets and backdoor/frontdoor criteria  
- Matching, weighting, and ML-based estimation  
- Hands-on examples and intuition-building graphical exercises

---

### **11:30–12:30 — Lunch Break**

---

### **12:30–13:45 — Counterfactuals & Algorithmic Counterfactuals**
- Counterfactual semantics in SCMs  
- Causal queries vs. statistical predictions  
- Individual-level vs. population-level counterfactuals  
- Algorithmic counterfactuals in ML systems  
- Applications in fairness, reasoning, simulation, and generative models

---

### **13:45–14:00 — Break**

---

### **14:00–15:30 — Causal Modeling in ML Systems**
- Causal inference with machine learning estimators  
- Causal forests, meta-learners (T-, S-, X-, R-learners)  
- Causal representation learning  
- Causal reasoning in reinforcement learning and decision-making systems  
- Case study: experimentation and adaptive policies

---

### **15:30–15:45 — Break**

---

### **15:45–17:00 — Deep Causal Models & Future Directions**
- Deep causal latent variable models  
- Causal generative models and structural VAEs  
- Counterfactual simulation using probabilistic programming  
- Causality for world models, safety, and alignment  
- A practitioner’s playbook:
  - How to choose the right causal tools  
  - How causal reasoning complements ML  
  - Pitfalls and anti-patterns  
  - Recommended reading and research roadmap  
- Open Q&A

---

## **Software and Tools**

While the workshop is conceptual and model-based, demonstrations use:

- **Pyro** for probabilistic programming and counterfactual simulation  
- Lightweight PyTorch examples for causal estimators and structural models  

(Hands-on notebooks may be provided depending on event format.)

---

## **Suggested Background Reading**

- *Causal AI* — Robert Osazuwa Ness  
- *Causal Inference in Statistics: A Primer* — Pearl, Glymour, Jewell  
- *Causality* — Judea Pearl  
- Key tutorials on SCMs, do-calculus, and ML-based causal inference

---

## **Instructor**

**Robert Osazuwa Ness** is a researcher specializing in causal AI, generative modeling, probabilistic programming, and counterfactual reasoning. He has worked as an AI research scientist in both industry and academia and is the author of *Causal AI*.

---
