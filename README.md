# Integrating Prior Knowledge Into Neural Networks Explicitly

This repository is home to three different methods:
1. **The Two-stage Method:** ["Prior knowledge meets Neural ODEs: a two-stage training method for improved explainability"; C. Coelho, M. Fernanda P. Costa, L.L. Ferrás; The First Tiny Papers Track at ICLR 2023](https://openreview.net/forum?id=p7sHcNt_tqo&referrer=%5Bthe%20profile%20of%20C.%20Coelho%5D(%2Fprofile%3Fid%3D~C._Coelho2)) and ["A Two-Stage Training Method for Modeling Constrained Systems with Neural Networks", C. Coelho, M. Fernanda P. Costa and L.L. Ferrás; preprint](https://arxiv.org/abs/2403.02730)
2. **The Self-adaptive Penalty Method:** ["A Self-Adaptive Penalty Method for Integrating Prior Knowledge Constraints into Neural ODEs"; C. Coelho, M. Fernanda P. Costa, and L.L. Ferrás; preprint](https://arxiv.org/abs/2307.14940) 
3. **The Filter Method:** ["A Filter-based Neural ODE Approach for Modelling Natural Systems with Prior Knowledge Constraints"; C. Coelho, M. Fernanda P. Costa, and L.L. Ferrás; Accepted to Knowledge-Guided Machine Learning Workshop at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases ECML PKDD 2023]()


## Usage Instructions

### **The Two-stage Method**


We introduced a two-stage training method for Neural ODEs aimed at explicitly incorporating prior knowledge constraints into the model.

The proposed two-stage method rewrites constrained optimisation problems as two unconstrained sub-problems, solving them sequentially during the Neural ODE optimisation process. In the first stage, the loss function is defined by the total constraints violations, to find a feasible solution of the original constrained problem. Subsequently, the second stage starts with the solution from the first stage and optimises a loss function given by the original loss function. To keep the optimisation process inside the feasible region during the second stage, a preference point strategy, featuring two variations, is proposed. This strategy rejects any point that is infeasible or does not improve admissibility, proceeding with either the point with the best admissibility value or the previous iteration point.

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


We proposed a self-adaptive penalty function and a self-adaptive penalty algorithm for Neural ODEs to enable modelling of constrained real-world systems. The proposed self-adaptive penalty algorithm can dynamically adjust the penalty parameters.

 Our approach is an improvement over traditional penalty methods which require an appropriate initialisation and tuning of penalty parameters $\boldsymbol{\mu}$ during the optimisation process. This selection is challenging and time-consuming. In general, in the context of NNs, a fixed penalty parameter is used, and consequently an optimal solution may not be found.
 
 The proposed self-adaptive penalty algorithm dynamically adjusts penalty parameters taking into account the degree of constraints violation, resulting in more efficient and accurate predictions, especially for complex systems with constraints. The self-adaptive penalty function employs a normalisation step to ensure all values (loss and constraints violations) have the same order of magnitude, improving training stability, critical for complex systems prone to numerical instability.

To run the **population growth** example, execute the following command, choosing a path to store the plot with the results and final trained model weights:

```
python selfAdaptive/logPopulationAdaptivePenalty.py --savePlot "path/to/save/plot" --saveModel "path/to/save/model"
```

To run the **chemical reaction** example, execute the following command, choosing a path to store the plot with the results and final trained model weights:

```
python selfAdaptive/chemicalReactionAdaptivePenalty.py --savePlot "path/to/save/plot" --saveModel "path/to/save/model"
```
To run the **oscillator** example, execute the following command, choosing a path to store the plot with the results and final trained model weights:

```
python selfAdaptive/dampedOscillatorSelfAdaptive.py --savePlot "path/to/save/plot" --saveModel "path/to/save/model"
```

The update strategy with the preference point can be turned on and off using the ```--updateStrategy``` flag.

### **The Filter Method**

With the filter method, we aim to minimise the constraints violation and the loss function simultaneously. At each iteration, the filter-based \ac{Neural ODE} method generates two independent trial points. One computed by minimising the loss function and another minimising the constraints violation. The proposed filter-based \ac{Neural ODE} method prioritises feasible solutions, followed by infeasible and non-dominated ones.

To run the **population growth** example, execute the following command, choosing a path to store the plot with the results and final trained model weights:

```
python filterMethod/logPopulation.py --savePlot "path/to/save/plot" --saveModel "path/to/save/model"
```

To run the **chemical reaction** example, execute the following command, choosing a path to store the plot with the results and final trained model weights:

```
python filterMethod/chemicalReaction.py --savePlot "path/to/save/plot" --saveModel "path/to/save/model"
```

