# Integrating Prior Knowledge Into Neural Networks Explicitly

This repository is home to three different methods:
1. **The Two-stage Method:** ["Prior knowledge meets Neural ODEs: a two-stage training method for improved explainability"; C. Coelho, M. Fernanda P. Costa, L.L. Ferrás; The First Tiny Papers Track at ICLR 2023](https://openreview.net/forum?id=p7sHcNt_tqo&referrer=%5Bthe%20profile%20of%20C.%20Coelho%5D(%2Fprofile%3Fid%3D~C._Coelho2)) and ["A Two-Stage Training Method for Modeling Constrained Systems with Neural Networks", C. Coelho, M. Fernanda P. Costa and L.L. Ferrás; preprint](https://arxiv.org/abs/2403.02730)
2. **The Self-adaptive Penalty Method:** ["A Self-Adaptive Penalty Method for Integrating Prior Knowledge Constraints into Neural ODEs"; C. Coelho, M. Fernanda P. Costa, and L.L. Ferrás; preprint](https://arxiv.org/abs/2307.14940) 
3. **The Filter Method:** ["A Filter-based Neural ODE Approach for Modelling Natural Systems with Prior Knowledge Constraints"; C. Coelho, M. Fernanda P. Costa, and L.L. Ferrás; Accepted to Knowledge-Guided Machine Learning Workshop at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases ECML PKDD 2023]()


## Usage Instructions

### **The Two-stage Method**

To run the **population growth** example, execute the following command, choosing a path to store the plot with the results and final trained model weights:

```
python 2Stage/logPopulation.py --savePlot "path/to/save/plot" --saveModel "path/to/save/model"
```

To run the **chemical reaction** example, execute the following command, choosing a path to store the plot with the results and final trained model weights:

```
python 2Stage/chemicalReaction.py --savePlot "path/to/save/plot" --saveModel "path/to/save/model"
```

There are more options that can be chosen, such as the number of iterations, data size, numerical method, etc. Check the options in the logPopulation.py and chemicalReaction.py files.

### **The Self-adaptive Penalty Method**

### **The Filter Method**
