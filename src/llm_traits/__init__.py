"""Run one trait-extraction methodology on many traits and compare the results.

The methodology is the one from Tagliabue, Dung and Berg, *The Pain Axis*
(arXiv:2609.16247): denoised difference-in-means over matched contrastive
sentence sets, with the read-out layer chosen by cross-validated projection
AUC, then validated by a steering coefficient ladder, a self-versus-user
scenario asymmetry, and a relief-button choice task.

Nothing here is an argument that the pain result is wrong. The package exists
because the methodology takes the trait as an argument, and a result that only
means something when the argument is "pain" is a result about the argument.
"""

__version__ = "0.1.0"
