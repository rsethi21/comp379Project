# Methods
## Resource
- https://superwise.ai/blog/dealing-with-machine-learning-bias/
## Compare different model architectures:
- All seem very similar in terms of performance
## Debiasing using Resampling:
- VAEs to balance the distribution of overall latent space of the input
    - Status: applied to neural network in the debiasing script
- Remove variables that are biased
    - Status: try on all models
- Create new data samples of less represented spaces
    - Status: not attempted but could use VAE or GAN to generate new training examples
- Adding Counterfactual Data
    - Status: not attempted but could do it for one variable and see the result
- Uncertainty output to inform user
    - try on all models
- Fairness contraints
    - try on all models (own implementations of loss function)