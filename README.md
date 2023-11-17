# comp379Project
## How to run experiment.py (brackets indicate an optional argument)
```
python3 experiment.py -i path/to/dataset -p number/of/processors/to/use [-o output/folder]
```
## Dataset:
- https://www.kaggle.com/datasets/reihanenamdari/breast-cancer
## Goal:
- Identify/Quantify biases of the dataset
- Examine how different model algorithms (SVM, KNN, NN, LR, etc.) deal with such bias
  - Understand how models are making decisions (permutation importance, gradient-based methods)
- Explore ways to address such biases (adjusted cross fold validation, bias penalties, etc.)
