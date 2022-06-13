---
layout: publication
type: journal
title:  "Inference of viral quasispecies with a paired de Bruijn graph"
authors: borja, susana, jose, leena
journal: "Bioinformatics"
shortjournal: ""
doi: 10.1093/bioinformatics/btaa782
date: 01-05-2021
highlighted: True
pubmed: false
abstract: |
   *Motivation*: RNA viruses exhibit a high mutation rate and thus they exist in infected cells as a population of closely related strains called viral quasispecies. The viral quasispecies assembly problem asks to characterize the quasispecies present in a sample from high-throughput sequencing data. We study the de novo version of the problem, where reference sequences of the quasispecies are not available. Current methods for assembling viral quasispecies are either based on overlap graphs or on de Bruijn graphs. Overlap graph-based methods tend to be accurate but slow, whereas de Bruijn graph-based methods are fast but less accurate.

   *Results*: We present viaDBG, which is a fast and accurate de Bruijn graph-based tool for de novo assembly of viral quasispecies. We first iteratively correct sequencing errors in the reads, which allows us to use large k-mers in the de Bruijn graph. To incorporate the paired-end information in the graph, we also adapt the paired de Bruijn graph for viral quasispecies assembly. These features enable the use of long-range information in contig construction without compromising the speed of de Bruijn graph-based approaches. Our experimental results show that viaDBG is both accurate and fast, whereas previous methods are either fast or accurate but not both. In particular, viaDBG has comparable or better accuracy than SAVAGE, while being at least nine times faster. Furthermore, the speed of viaDBG is comparable to PEHaplo but viaDBG is able to retrieve also low abundance quasispecies, which are often missed by PEHaplo.

   *Availability and implementation*: viaDBG is implemented in C++ and it is publicly available at https://bitbucket.org/bfreirec1/viadbg. All datasets used in this article are publicly available at https://bitbucket.org/bfreirec1/data-viadbg/.

---

## Abstract

{{ page.abstract }}
