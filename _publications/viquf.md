---
layout: publication
type: journal
title:  "ViQUF: de novo Viral Quasispecies reconstruction using Unitig-based Flow networks"
authors: borja, susana, jose, leena
journal: "BMC (Submitted)"
shortjournal: ""
doi: https://arxiv.org/abs/2112.06590
date: 12-12-2021
openaccess: true
pubmed: false
highlighted: true
abstract: |
    During viral infection, intrahost mutation and recombination can lead to significant evolution, resulting in a population of viruses that harbor multiple haplotypes. The task of reconstructing these haplotypes from short-read sequencing data is called viral quasispecies assembly, and it can be categorized as a multiassembly problem. We consider the de novo version of the problem, where no reference is available. 
    We present ViQUF, a de novo viral quasispecies assembler that addresses haplotype assembly and quantification. ViQUF obtains a first draft of the assembly graph from a de Bruijn graph. Then, solving a min-cost flow over a flow network built for each pair of adjacent vertices based on their paired-end information creates an approximate paired assembly graph with suggested frequency values as edge labels, which is the first frequency estimation. Then, original haplotypes are obtained through a greedy path reconstruction guided by a min-cost flow solution in the approximate paired assembly graph. ViQUF outputs the contigs with their frequency estimations. Results on real and simulated data show that ViQUF is at least four times faster using at most half of the memory than previous methods, while maintaining, and in some cases outperforming, the high quality of assembly and frequency estimation of overlap graph-based methodologies, which are known to be more accurate but slower than the de Bruijn graph-based approaches.
    * Code: * https://github.com/borjaf696/ViQUF
---

## Abstract

{{page.abstract}}
