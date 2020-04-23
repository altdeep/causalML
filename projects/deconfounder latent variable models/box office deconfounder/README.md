# Generative Model Implementation of the Causal Deconfounder for Box Office Predictions

## Authors 

[Siddesh Acharekar](https://www.linkedin.com/in/siddhesha/), Robert Ness

## Abstract

https://arxiv.org/pdf/1805.06826.pdf

This work implements the prediction of box office revenues given actors' presence in a film as described in Wang and Blei, 2019.  The problem is that if one uses machine learning to predict box office revenue given actors, those predictions would have problems if used to then select who to cast in a new film.  Selecting cast members for a new film would be an intervention.  The intervention distribution of box office revenue is different from the observational distribution because of confounders.  For example, the presence of the actress [Cobie Smulders](https://www.imdb.com/name/nm1130627/) is correlated with high revenue, though common sense suggests she does not drive these high revenues.  This is because the causal effect of her presence on revenue is confounded by whether or not a film is a Marvel Comics film (Many Marvel films have cast Smulders, and Marvel films tend to be high earners at the box office).  The deconfounder method attempts to use the latent variables in latent variable generative models to block this back door path.  

Wang and Blei's original approach train a latent variable model, augment the data with estimates of latent variable values, then use the augmented data to predict the box office outcome.  In this work, we implement this approach as a fully generative model in Pyro.  This greatly simplifies the procedure and provides additional insights into modeling.  We show examples of adjusting for the "Marvelness" of a film when making decisions about Cobie Smulders.

## References

* Wang, Yixin, and David M. Blei. "[The Blessings of Multiple Causes.](https://arxiv.org/pdf/1805.06826.pdf)" Journal of the American Statistical Association(2019): 1-71.
