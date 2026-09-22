"""Model architectures, training loops, uncertainty quantification and evaluation.

Uncertainty is first-class here: predictors return a calibrated uncertainty estimate
alongside each prediction (MC dropout / deep ensembles).
"""
