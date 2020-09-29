# Causal Scene Generation

![sprites](https://miro.medium.com/max/1400/0*lacOtDtTlOQ9NYlp)

This is project work by [Harish Ramani](https://www.linkedin.com/in/harishramani1792).

This article summarizes a project integrating a causal directed graphical model with a variational autoencoder (VAE). Deep generative models like VAEs can generate synthetic images that are comparable to a set of training images.

The content of the images in the training data follow rules. At the most basic level, they follow the laws of physics. However, within a specific domain, the rules can simplify to set of abstract entities and relationships between entities. For example, your training data could have a simple set of entities like “boy”, “girl”, and “dog”, and a set of interactions including “boy walks dog” and “dog bites girl” and “girl greets boy.”

Suppose you wanted to specify these rules when generating from the model? For example, you might want to see what an image looks like if “boy greets boy.” You could add examples of those interactions to your training data. What if it were “girl bites dog”? It might be challenging to find such an image for inclusion in the training data if working with real-world images.

One solution is to build a deep causal generative model. The entities and the rules of their interactions come from knowing the domain you wish to model and are made explicit in the model architecture. The mapping of those entities and relationships to pixels in an image is up to the neural network elements of the model architecture — in the case of a VAE, this is the decoder. Given an image, the encoder maps us back to the abstractions describing the entities and interactions between entities characterized within the image, and we can reason about them causally.

This tutorial demonstrates proof of concept for this flavor of modeling using procedurally generating images of fights between characters one might find in a role-playing video game.

The full tutorial is published as an [online book](https://linkinnation1792.gitbook.io/causal-scene-generation/
).  You can also check out the [10 minute summary blog post](https://medium.com/@ramani.h/728d3450b600).
