# Good-Action Identification
Code for *Lenient Regret and Good-Action Identification in Gaussian Process Bandits*

Copyright by the authors



## Dependencies

-	Python 3
-	NumPy
-	SciPy 
-	Scikit-Learn
-	Matplotlib
-	GPy
-	PyTorch (for quasi-random sequences)
-	pybox2d & pygame (for robot pushing)
-	xgboost (for XGBoost)




## The noisy/noiseless experiment on synthetic/real-world functions
*Input* arguments for `main.py`:

- 	**`function`**:  Specify the function name; See `good_action/utils.py` for details
- 	**`noisy`**: Noisy or noiseless observation
-	 **`epsilon`**:  The good-action threshold; Float value

*Output*:

- 	**`log file`**:  Running logs
- 	**`query histories`**: .npy file saving queried points and values

*For example*:

* Testing on the noiseless 3D robot pushing function
```python
python main.py robot3 noiseless 4.5
```

*Visualization*: Run `plot.ipynb`



## The lenient regret experiment on synthetic GP function

*Input* arguments for `lenient.py`:

- 	**`epsilon`**:  The good-action threshold; Float value; Default=0.9

*Output*:

- 	**`lenient and standard regrets`**:  .npy file

*For example*:

```python
python lenient.py 0.9
```

*Visualization*: Run `plot_lenient.ipynb`
