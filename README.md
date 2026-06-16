# Generalized mi-KSPSVM via Threshold-Based Assumption

MATLAB implementation of the **Generalized Kernelized Semiproximal SVM for Multiple Instance Learning**, proposed as an extension of Avolio & Fuduli (2024).

## Overview

The generalized mi-KSPSVM extends the original mi-KSPSVM to the **threshold-based MIL assumption**, requiring at least $r \geq 1$ positive instances per positive bag. This makes the classifier more robust to noise and outliers. When $r=1$, the algorithm reduces exactly to the original mi-KSPSVM.

## Requirements

- MATLAB R2018a or later
- Optimization Toolbox (`quadprog`)

## Usage

1. Download the benchmark MIL datasets from: https://www.cs.columbia.edu/~andrews/mil/data/

2. Place the `.mat` files directly in a `datasets/` folder.
   
3. Update the `data_dir` variable in `Gen_mi_KSPSVM.m` to point to your `datasets/` folder:
```matlab
   data_dir = 'path/to/your/datasets/';
```

4. Run the algorithm on a dataset:
```matlab
   Gen_mi_KSPSVM('elephant')
   Gen_mi_KSPSVM('tst1')
   Gen_mi_KSPSVM('musk1')
```

5. Results are saved automatically to a `.txt` file in `data_dir`.

## Algorithm Details

The algorithm runs for $r \in \{1, 2, 3\}$ on each dataset using:

- **Outer CV:** 10-fold cross-validation for performance evaluation
- **Inner CV:** 5-fold cross-validation for hyperparameter selection
- **Hyperparameter grid:** $C \in \{2^i \mid i=-7,\ldots,7\}$, $\sigma \in \{2^{-3}, 2^{-2}, 2^2, 2^3, 2^4\}$
- **Kernel:** RBF kernel
- **Preprocessing:** TruncatedSVD (200 components) applied automatically for high-dimensional datasets (TST series)

## Output

For each dataset, a results file is generated containing:
- Best hyperparameters per fold
- Accuracy per fold
- Final average: Accuracy, Sensitivity, Specificity, F-score, CPU time

## Reference

M. Avolio and A. Fuduli, "A kernelized semiproximal support vector machine for multiple instance learning," *Optimization Letters*, 18, 635–649, 2024.

The datasets were originally introduced in:
S. Andrews, I. Tsochantaridis, and T. Hofmann, "Support vector machines for multiple-instance learning," *Advances in Neural Information Processing Systems*, 2003.
