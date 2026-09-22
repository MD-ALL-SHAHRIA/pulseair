"""CTGAN-based synthetic data augmentation (via ``sdv``).

Trains a conditional tabular GAN on the real sensor records, samples synthetic rows to
rebalance rare high-pollution regimes, and scores synthetic-data quality before it is
allowed into a training set.
"""
