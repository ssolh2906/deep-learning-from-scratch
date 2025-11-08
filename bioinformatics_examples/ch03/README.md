# Splice Position Weight Matrix (PWM)
3-class classification problem

### Splice site
Where intron starts, or end
Donor site(5' splice site): End of an Exon, beginning of an intron
Acceptor site(3' splice site): End of an Intron, beginning of an exon

### PMW(Position Weight Matrix)
The matrix include patterns of splice sites
Based on this matrix, model can score bases for each classes EI,IE and N

## Dataset
Splice-junction gene sequences (UCI dataset)
60-mer
label: EI(Donor), IE(Acceptor), N(Neither of them)

### Molecular Biology (Splice-junction Gene Sequences)
**Source**: [UCI Machine Learning Repository — Splice‑junction gene sequences](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences)  
**Number of instances**: 3190  
**Number of Features**: 60 (each feature is a nucleotide A,C,G,T)  
**DOI**: 10.24432/C5M888  
