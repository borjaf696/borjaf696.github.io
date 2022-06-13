---
layout: publication
type: journal
title:  "Memory-Efficient Assembly using Flye"
authors: borja, susana, jose
journal: "IEEE/ACM Trans Comput Biol Bioinform"
shortjournal: ""
doi: 10.1109/TCBB.2021.3108843
date: 01-09-2021
openaccess: true
pubmed: 32272874
highlighted: true
abstract: |
    In the past decade, next-generation sequencing (NGS) enabled the generation of genomic data in a cost-effective, high-throughput manner. The most recent third-generation sequencing technologies produce longer reads; however, their error rates are much higher, which complicates the assembly process. This generates time- and space- demanding long-read assemblers. Moreover, the advances in these technologies have allowed portable and real-time DNA sequencing, enabling in-field analysis. In these scenarios, it becomes crucial to have more efficient solutions that can be executed in computers or mobile devices with minimum hardware requirements. We re-implemented an existing assembler devoted for long reads, more concretely Flye, using compressed data structures. We then compare our version with the original software using real datasets, and evaluate their performance in terms of memory requirements, execution speed, and energy consumption. The assembly results are not affected, as the core of the algorithm is maintained, but the usage of advanced compact data structures leads to improvements in memory consumption that range from 22% to 47% less space, and in the processing time, which range from being on a par up to decreases of 25%. These improvements also cause reductions in energy consumption of around 3-8%, with some datasets obtaining decreases up to 26%.

---

## Abstract

{{page.abstract}}
