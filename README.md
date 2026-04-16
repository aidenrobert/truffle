# TRUFFLE:
**T**rained **R**ecognition of **U**nique **F**luorescence- & **F**orm-based **L**abels for **E**nvironmental aerosols.

Classification algorithm code made for the multiparameter bioaerosol spectrometer (MBS; University of Hertfordshire, UK). This code was produced and outlined in Jönsson et al. (*in review*). It is a machine learning-based particle classifier that aims to:

1. Identify and flag likely non-biological interferent fluorescent particles with a **pollution model.**
2. Classify fluorescent particles as belonging to a pollution/combustion, pollen, bacteria, or fungal spore class with a **multiclass classifier.**
3. Identify non-fluorescent coarse-mode particles as being either sea spray aerosol (SSA)-like or dust-like with a **dust model.**

The details of its training can be found in:

Jönsson, A., Fu, J., Freitas, G. P., Crawford, I., Dagsson-Waldhauserová, P., Krejci, R., Tobo, Y., Yttri, K. E., & Zieger, P. (2026). **Tracing biological, human, and inorganic sources of coarse aerosols via single-particle fluorescence and optical morphology.** *EGUsphere* (in review at *Atmospheric Chemistry & Physics*, 2026, 1-37.

## Main functions

### `truffle.flag_pollution`:
This component is a logistic regression model (LRM) trained on fluorescent particles. It will apply the LRM to predict the likelihood of a fluorescent particle being a non-biological, interfering combustion particle, based on laboratory data of bioaerosol characterizations and observations of strong combustion events.

### `truffle.classify_fluo`:
This component is a multiclass classifier based on dimension reduction using uniform manifold approximation and projection (UMAP) transformation and a $k$-nearest neighbors (kNN) classifier. It will base the classification on similarities to laboratory characterization data of bioaerosols and combustion particles, and assign a likely class among the following to each fluorescent particle:
- 🌼 **Pollen:** These particles are likely pollen fragments (not intact pollen; the MBS's size range does not allow it to measure most whole pollen grains), and have broad, varied fluorescence curves.
- 🦠 **Bacteria:** These particles have a distinct signal of strong fluorescence in the *B* detection channel, attributable to tryptophan.
- 🍄‍🟫 **Fungal spores:** These particles share some similarities with bacteria, but have varied fluorescence peaks that can be either in the *B* or *C* channels.
- 🔥 **Combustion:** These particles share similarities with pollen particles, but typically have stronger fluorescence.

### `truffle.flag_dust`:
This component is a LRM trained on laboratory-characterized non-fluorescent particles and quantifies the probability of being either SSA or dust.

### `truffle.pig`🐷:
**P**article **I**dentification **G**adget. This combines and calls all of the component models of the classification algorithm and applies them to an input dataset.

## Other utilities

### `truffle.preprocess`:
Preprocesses raw MBS data files in the way that chooses the input parameters needed for the classification algorithm, and other parameters useful for analysis. This routine also includes the decision tree method, and code, developed by Gabriel Freitas for [Freitas et al. (2022)](https://pubs.rsc.org/en/content/articlelanding/2022/ea/d2ea00047d).

### `truffle.concentration`:
Calculates concentrations with a chosen time resolution for a given dataframe of particles. This routine is also based on code produced by Gabriel Freitas, hosted at [https://github.com/SU-air/instrumentation-MBS](https://github.com/SU-air/instrumentation-MBS).
