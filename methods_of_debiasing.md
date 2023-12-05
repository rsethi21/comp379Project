# Methods
## Resource
- https://superwise.ai/blog/dealing-with-machine-learning-bias/
- https://ocw.mit.edu/courses/res-ec-001-exploring-fairness-in-machine-learning-for-international-development-spring-2020/pages/module-four-case-studies/case-study-mitigating-gender-bias/
## Compare different model architectures:
- Status: Tried on all models
- All seem very similar in terms of performance, seems ensemble performs much better
## Debiasing using Data-based Methods:
- Resampling multiple variables
    - Method: VAEs to balance the distribution of overall latent space of the input
    - Status: applied to neural network in the debiasing_vae script
- Ignorance
    - Method: Remove variables that are biased from the dataset (examine the distribution of each of the following variables: AnyHlthCare, Gender, Income, Education, Age) and remove the most problamatic one
    - Status: need to try on all models
- Create new data samples
    - Method: Training a model to generate new datapoints with respect to a certain biased attribute (GANs, NB, or other generative ai methods)
    - Status: need to try on all models
- Adding counterfactual data
    - Method: keep all other variables same except the biased attribute and add this new false entry to the dataset in such a way that it balances that attribute (easier to use with a binary variable)
    - Status: need to try on all models
- Uncertainty output to inform user
    - Method: I'm not sure but there are many resources or models that can give uncertainty scores
    - Status: try on all models
- Fairness contraints
    - Method: create own custom loss function that forces a balancing of performance for certain attributes
    - Status: try on all models; applied to neural networks in hyperparameter selection in combination with the resampling method whcih seems to improve performance across the income categories in terms of balancing
