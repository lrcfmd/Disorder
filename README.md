# Disorder

The repository contains scripts and supporting files which were used to process data in the paper "Structural disorder in crystalline materials" (link to the preprint). In particular it allows:

(1) extract disorder information from CIF and assign a to each crystallographic orbit in a compound one of the labels: O (ordered), S (subsitutional disorder), V (vacancies), P(positional disorder), SV (combination of S and V in one orbit), SP, VP, and SVP. There are also COM orbits in complicated cases, which cannot be processed by our classification scheme (the fraction of such compounds among known compounds is small). 

(2) Calculate mixing and configurational entropy of crystalline material from CIF. The difference between mixing and configurational entropies is that configurational entropy has contribution from positional disorder, while mixing does not.

(3) Code which was used to generate all figures in the paper is provides.
