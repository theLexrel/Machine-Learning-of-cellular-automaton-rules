# Machine-Learning-of-cellular-automaton-rules
Code written for the master thesis 'Machine Learning of cellular automaton rules' by Max Müller at TU Dresden, 2022/23

## Use of the code in the master thesis
The files were used for the following purpose in the master thesis:
- CA_1D.py
  - Thesis_visuals(): The examples of the CA time series in section 2.2, Cellular Automaton - CA in 1D
  - Thesis_proveOfConcept(): The figures, extracted transition probabilities and calculated error measures in section 5.2.1, Rule extraction from a CA time series - Proof of concept for CA in 1D 
  - Thesis_realData(): The figures, extracted transition probabilities and calculated error measures in section 5.3.1, Rule extraction from a CA time series - Simulation of real data in 1D 
- CA_2D.py 
  - Thesis_visuals(): The examples of the CA time series in section 2.3, Cellular Automaton - CA in 2D
  - Thesis_proveOfConcept(): The figures, extracted transition probabilities and calculated error measures in section 5.2.2, Rule extraction from a CA time series - Proof of concept for CA in 2D 
  - Thesis_Thesis_real_correctNeighborhood(): The figures, extracted transition probabilities and calculated error measures for a correct match between hypothesis implementation and neighborhood used by the update rule in section 5.3.2, Rule extraction from a CA time series - Simulation of real data in 2D
  - Thesis_Thesis_real_wrongNeighborhood(): The figures, extracted transition probabilities and calculated error measures for a correct mismatch between hypothesis implementation and neighborhood used by the update rule in section 5.3.2, Rule extraction from a CA time series - Simulation of real data in 2D  - LGCA_1D.py
  - Thesis_ExtractionInteractionStrenght(): The figures and extracted coefficients for the LGCA with a 1D lattice in section 6.2, Rule extraction from LGCA time series - Extracting the interaction parameters 
  - Thesis_GeneralApproach(): The figures for the LGCA with a 1D lattice in section 6.3, Rule extraction from LGCA time series - Principle approach to building the hypothesis 
- LGCA_2D.py:
  - main(): builds figures used in section 6.2 and 6.3 for the LGCA with a 2D lattice and gives out the corresponding coefficients 
  
## Required packages
The python code will only run with the packages *matpltlib*, *numy* and *sklearn* with CA_2D.py additionally requiring *numba* and LGCA_1D.py & LGCA_2D.py needing the external software package *BIO-LGCA*. The last one was introduced in the paper *'BIO-LGCA: A cellular automaton modelling class for analysing collective cell migration'* by Deutsch et al, published in PLOS Computational Biology in June 2021 (<a href=https://doi.org/10.1371/journal.pcbi.1009066> link </a>), which is accessible on <a href=https://github.com/sisyga/BIO-LGCA>github</a>.

## General comments on the code
A lot of improvemnets could have been made to the code... 
