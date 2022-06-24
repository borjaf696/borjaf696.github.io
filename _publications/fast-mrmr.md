---
layout: publication
type: journal
title:  "Parallel Feature Selection for Distributed-Memory Clusters"
authors: borja,jorge
journal: "Information Sciences"
shortjournal: ""
doi: 10.1016/j.ins.2019.01.050
date: 29-10-2018
software_res: https://github.com/borjaf696/Fast-mRMR
openaccess: true
highlighted: true
pubmed: false
abstract: |
    Feature selection is nowadays an extremely important data mining stage in the field of machine learning due to the appearance of problems of high dimensionality. In the
literature there are numerous feature selection methods, mRMR (minimum-RedundancyMaximum-Relevance) being one of the most widely used. However, although it achieves good results in selecting relevant features, it is impractical for datasets with thousands of features. A possible solution to this limitation is the use of the fast-mRMR method, a greedy optimization of the mRMR algorithm that improves both scalability and efficiency. In this work we present fast-mRMR-MPI, a novel hybrid parallel implementation that uses MPI and
OpenMP to accelerate feature selection on distributed-memory clusters. Our performance evaluation on two different systems using five representative input datasets shows that fastmRMR-MPI is significantly faster than fast-mRMR while providing the same results. As an example, our tool needs less than one minute to select 200 features of a dataset with more
than four million features and 16,000 samples on a cluster with 32 nodes (768 cores in total), while the sequential fast-mRMR required more than eight hours. Moreover, fast-mRMRMPI distributes data so that it is able to exploit the memory available on different nodes of a cluster and then complete analyses that fail on a single node due to memory constraints.
Our tool is publicly available at https://github.com/borjaf696/Fast-mRMR.
---

## Abstract

{{ page.abstract }}
