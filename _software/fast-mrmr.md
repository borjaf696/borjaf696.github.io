---
title: fast-mRMR
type: Software
external_url: https://github.com/borjaf696/Fast-mRMR
date: 
highlighted: true
---

Several optimizations have been introduced in this improved version in order to speed up the costliest computation of the original algorithm: Mutual Information (MI) calculations. These optimizations are described in the followings: 

- **Cache marginal probabilities**: Instead of obtaining the marginal probabilities in each MI computation, those are calculated only once at the beginning of the program, and cached to be re-used in the next iterations.

- **Accumulating redundancy**: The most important optimization is the greedy nature of the algorithm. Instead of computing the mutual information between every pair of features, now redundancy is accumulated in each iteration and the computations are performed between the last selected feature in S and each feature in non-selected set of attributes. 

- **Data access pattern**: The access pattern of mRMR to the dataset is thought to be feature-wise, in contrast to many other ML (machine learning) algorithms, in which access pattern is row-wise. Although being a low-level technical nuance, this aspect can significantly degrade mRMR performance since random access has a much greater cost than block-wise access.

Fast-mRMR-MPI version:
- **Sequential**: original fast-mRMR code for develop features selection.
- **OpenMP**: OpenMP approach allows fast-mRMR to be applied by people or research groups with low resources which leads an acceleration of the original fast-mRMR code scaling with the number of cores.
- **MPI**: MPI approach allows fast-mRMR to be applied over big data datasets in cluster systems.