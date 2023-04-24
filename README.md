# HOOK
HOOK is an Anti-Recommender Recommender. HOOK fishes for music recommendations from the space between typical music recommendation spaces.

The SampleCNN used to produce musical representations in HOOK is trained by Contrastive Learning for Musical Representation[(CLMR)](https://arxiv.org/pdf/2103.09410.pdf), a [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) based training technique developed by Janne Spijkervet and John Ashley Burgoyne. The HOOK SampleCNN itself is a modified version of the CLMR pre-trained SampleCNN implementation found [here](https://github.com/Spijkervet/CLMR), as is most of it's supporting code derived from the CLMR project.
