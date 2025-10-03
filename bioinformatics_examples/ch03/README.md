# Splice Position Weight Matrix (PWM)
3-class classification problem

### Splice site
Where intron starts, or end
Donor site(5' splice site): End of an Exon, beginning of an intron
Acceptor site(3' splice site): End of an Intron, beginning of an exon

### Dataset
Splice-junction gene sequences (UCI dataset)
60-mer
label: EI(Donor), IE(Acceptor), N(Neither of them)

### PMW(Position Weight Matrix)
The matrix include patterns of splice sites
Based on this matrix, model can score bases for each classes EI,IE and N

