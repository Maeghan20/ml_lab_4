# Viva Cheat Sheet: PCA vs LDA vs ICA (Lab 4)

## 30-second summary
- Data: 569 samples x 30 features, 2 classes (212 malignant, 357 benign). Standardised first.
- PCA: 2 PCs keep 63.2% variance; 10 PCs keep 95.2%; 17 keep 99%.
- LDA: only 1 component possible (classes - 1). Best separation: silhouette 0.68, CV accuracy ~96.8%.
- ICA: 2 independent components, uncorrelated, non-Gaussian (kurtosis ~2.9 and ~2.5). Weakest separation (silhouette 0.37, ~94.0%).
- All 30 features with logistic regression: ~97.4% (baseline).

## Likely questions and short answers
1. **Why standardise?** PCA/ICA are scale-sensitive; `area` would dominate `smoothness`. Standardising gives each feature mean 0, std 1.
2. **Why only 1 LDA component?** Rank of the between-class scatter matrix is at most C-1 = 1.
3. **PCA vs LDA in one line?** PCA = direction of most variance (ignores labels). LDA = direction that best separates class means relative to spread (uses labels).
4. **Is independent the same as uncorrelated?** No. Independent implies uncorrelated, not the other way round. PCA only gives uncorrelated; ICA uses higher-order statistics for independence.
5. **Why does ICA need non-Gaussian data?** Gaussian rotations are indistinguishable (rotation-invariant), so sources can't be identified. Also CLT: mixtures are more Gaussian than sources.
6. **Why is ICA poor here?** The features aren't mixtures of independent sources, and independence has nothing to do with class labels. (Shown in self-learning E: ICA recovers mixed signals at ~0.998 correlation, PCA only ~0.93-0.96.)
7. **ICA ordering/sign?** Arbitrary. Components can be permuted, flipped, rescaled.
8. **How to choose number of PCA components?** Cumulative variance threshold (e.g. 95%), scree-plot elbow, or cross-validated downstream accuracy (self-learning A).
9. **What is the data-leakage point?** Supervised reducers (LDA) must be fitted only on the training fold. Self-learning D shows leaky CV is optimistic (0.975 vs 0.968).
10. **Limitations?** All linear. LDA assumes Gaussian classes with shared covariance. For non-linear structure use Kernel PCA / t-SNE / UMAP.
11. **What does PC1 mean?** Same-sign loadings on concavity, concave points, compactness, radius and perimeter: an overall size + shape-irregularity axis.
12. **Applications?** PCA: compression, visualisation, gene expression. LDA: face recognition (Fisherfaces), diagnosis, credit scoring. ICA: EEG artefact removal, cocktail party, fMRI.
