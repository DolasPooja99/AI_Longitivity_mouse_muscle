# Mouse Muscle Aging Atlas - Processing Pipeline

This repository contains a preprocessing script for converting single-nucleus RNA-seq (snRNA-seq) data from the **Mouse Muscle Aging Atlas** into `.parquet` files suitable for machine learning and visualization workflows.

Dataset Reference:[View on Hugging Face](https://huggingface.co/datasets/longevity-db/mouse-muscle-aging-atlas-snRNAseq)

---

## About the Dataset

The **Mouse Muscle Aging Atlas** provides single-nucleus transcriptomic data of skeletal muscle from mice across different ages and experimental conditions. This dataset allows researchers to:

- Study age-related cellular changes in mouse muscle
- Compare cell-type composition across genotypes or treatments
- Examine gene expression variation related to aging and longevity

The dataset includes:
- Normalized expression matrices
- Cell metadata (age, genotype, batch, etc.)
- Cell-type annotations
- Precomputed embeddings (optional)

---

## Script Overview

The `processing.py` script performs the following:

1. **Reads** the `.h5ad` AnnData file.
2. **Extracts** and saves:
   - `expression.parquet`: Expression matrix (cells × genes)
   - `gene_metadata.parquet`: Gene metadata
   - `cell_metadata.parquet`: Cell metadata
3. **Computes** or reuses precomputed:
   - PCA embeddings
   - UMAP embeddings
4. **Calculates**:
   - Highly Variable Genes (HVGs)
   - Gene statistics (mean expression, frequency)
   - Cell type proportions (overall and by donor/age)
   - Donor-level metadata summaries

All outputs are stored in: `mouse_muscle_aging_atlas_ml_data/`.

---

## Output Files

| File Name                                      | Description |
|-----------------------------------------------|-------------|
| `expression.parquet`                          | Dense gene expression matrix |
| `cell_metadata.parquet`                       | Metadata for each nucleus (e.g., age, type, batch) |
| `gene_metadata.parquet`                       | Annotations per gene |
| `pca_embeddings.parquet`                      | PCA-transformed coordinates |
| `umap_embeddings.parquet`                     | UMAP 2D coordinates |
| `pca_explained_variance.parquet`              | Variance explained by PCA |
| `highly_variable_gene_metadata.parquet`       | Subset of highly variable genes used in DR |
| `gene_statistics.parquet`                     | Summary stats for gene expression |
| `cell_type_proportions_overall.parquet`       | Cell type composition for the whole dataset |
| `cell_type_proportions_by_<group>.parquet`    | Grouped cell type proportions |
| `donor_metadata.parquet`                      | Aggregated metadata by mouse ID or sample ID |

---

## Requirements

Install the following Python packages:

```bash
pip install pandas numpy scanpy anndata scikit-learn umap-learn pyarrow
```

---

## Usage

Update this line in `processing.py`:

```python
H5AD_FILE_PATH = "//path/to/your/file.h5ad"
```

Then run:

```bash
python processing.py
```

All outputs will be saved under `mouse_muscle_aging_atlas_ml_data/`.

---

## Notes

- The script optionally computes cell cycle scores if gene lists are provided.
- Mitochondrial QC metrics can be computed using gene prefix filters (`mt-`, `Mt-`).
- Works even if PCA/UMAP are precomputed in the `.h5ad` file.


# Reference
- Source Dataset: https://www.muscleageingcellatlas.org/mouse-pp/

## Final Output

Final Dataset:  [Hugging Face dataset page](https://huggingface.co/datasets/longevity-db/mouse-muscle-aging-atlas-snRNAseq).


