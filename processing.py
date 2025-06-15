import pandas as pd
import anndata as ad
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import numpy as np
import scanpy as sc # Essential for single-cell specific tasks like HVG

# --- Configuration ---
# IMPORTANT: Replace with the actual path to your downloaded Mouse Muscle Atlas .h5ad file
H5AD_FILE_PATH = "/path/to/your/file.h5ad" # <<<--- CHANGE THIS PATH!

# Name of the directory to store the output Parquet files for mouse muscle
OUTPUT_DIR_NAME = "mouse_muscle_aging_atlas_ml_data"

# PCA Configuration
N_PCA_COMPONENTS = 50 # Number of principal components to compute. Adjust as needed.
APPLY_SCALING_BEFORE_PCA = True # Set to True to scale data (mean=0, std=1) before PCA

# UMAP Configuration
N_UMAP_COMPONENTS = 2 # Usually 2 or 3 for visualization
UMAP_N_NEIGHBORS = 15 # Number of nearest neighbors for UMAP. Adjust based on dataset size.
UMAP_MIN_DIST = 0.1 # Minimum distance for UMAP.

# HVG Configuration
N_TOP_HVGS = 4000 # Number of highly variable genes to select for dimensionality reduction. Adjust as needed.
# If you notice issues or want more/fewer features.

# Cell Cycle Gene Lists (for mouse) - IMPORTANT: You'll need actual lists if using
# These are often part of scanpy example data or found in public repositories.
# For simplicity, this script will demonstrate the *call*, but actual genes would be needed.
# For mouse, these are often capitalized (e.g., 'Mcm5', 'Pcna')
S_GENES_MOUSE = [] # Example: ['Mcm5', 'Pcna', 'Mki67'] - Placeholder!
G2M_GENES_MOUSE = [] # Example: ['Ccnb1', 'Cdk1', 'Top2a'] - Placeholder!
PERFORM_CELL_CYCLE_SCORING = False # Set to True if you have the gene lists and want to apply it

# --- 1. Create Output Directory ---
os.makedirs(OUTPUT_DIR_NAME, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR_NAME}")

# --- 2. Load the H5AD file into an AnnData object ---
try:
    adata = ad.read_h5ad(H5AD_FILE_PATH)
    print(f"Successfully loaded AnnData object from: {H5AD_FILE_PATH}")
    print(f"AnnData object shape: {adata.shape} (Cells x Genes)")
    print(f"Available layers: {list(adata.layers.keys())}")
    print(f"Available obsm keys: {list(adata.obsm.keys())}")
    print(f"Available varm keys: {list(adata.varm.keys())}")
    print(f"Available uns keys: {list(adata.uns.keys())}")

    # Ensure .X is dense for easier processing later if it's sparse
    if hasattr(adata.X, 'toarray'):
        adata.X = adata.X.toarray()
        print("Converted adata.X from sparse to dense array.")

    # Check for pre-computed embeddings
    perform_pca_flag = True
    if 'X_pca' in adata.obsm.keys() and N_PCA_COMPONENTS <= adata.obsm['X_pca'].shape[1]:
        print(f"\nFound pre-computed PCA in adata.obsm['X_pca']. Will use it.")
        perform_pca_flag = False
        precomputed_pca = adata.obsm['X_pca'][:, :N_PCA_COMPONENTS]

    perform_umap_flag = True
    if 'X_umap' in adata.obsm.keys() and N_UMAP_COMPONENTS <= adata.obsm['X_umap'].shape[1]:
        print(f"Found pre-computed UMAP in adata.obsm['X_umap']. Will use it.")
        perform_umap_flag = False
        precomputed_umap = adata.obsm['X_umap'][:, :N_UMAP_COMPONENTS]

except FileNotFoundError:
    print(f"Error: H5AD file not found at {H5AD_FILE_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the H5AD file: {e}")
    exit()

# --- Preprocessing & QC (Mouse Specific) ---

# Optional: Add mitochondrial gene percentage to adata.obs
# Common gene name pattern for mitochondrial genes in mouse: starts with 'mt-' or 'Mt-'
# You might need to adjust this pattern based on the gene IDs/names in your specific dataset.
# Example: If your gene names are like 'mt-Nd1', use 'mt-'. If 'MT-ND1', use 'MT-'.
adata.var['mt'] = adata.var_names.str.startswith('mt-') | adata.var_names.str.startswith('Mt-') # Adjust prefix if needed
if adata.var['mt'].any():
    print("Calculating mitochondrial gene content...")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # The result 'pct_counts_mt' will be in adata.obs

# Optional: Cell Cycle Scoring
if PERFORM_CELL_CYCLE_SCORING and S_GENES_MOUSE and G2M_GENES_MOUSE:
    print("Performing cell cycle scoring...")
    # Ensure raw counts or normalized counts are in adata.X before scoring
    # This assumes gene names are in adata.var_names and match S_GENES_MOUSE/G2M_GENES_MOUSE
    # sc.tl.score_genes_cell_cycle requires gene names as input
    try:
        sc.tl.score_genes_cell_cycle(adata, s_genes=S_GENES_MOUSE, g2m_genes=G2M_GENES_MOUSE)
        print("Cell cycle scores (S_score, G2M_score) and phase added to adata.obs.")
    except Exception as e:
        print(f"Warning: Could not perform cell cycle scoring. Check gene lists and data format. Error: {e}")
else:
    print("Skipping cell cycle scoring (PERFORM_CELL_CYCLE_SCORING is False or gene lists are empty).")


# --- 3. Save adata.X (Expression Matrix) as expression.parquet ---
expression_parquet_path = os.path.join(OUTPUT_DIR_NAME, "expression.parquet")
df_expression = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
df_expression.index.name = "cell_id"
df_expression.to_parquet(expression_parquet_path, index=True)
print(f"Saved expression data to: {expression_parquet_path}")

# --- 4. Save adata.var (Feature/Gene Metadata) as gene_metadata.parquet ---
gene_metadata_parquet_path = os.path.join(OUTPUT_DIR_NAME, "gene_metadata.parquet")
adata.var.index.name = "gene_id"
adata.var.to_parquet(gene_metadata_parquet_path, index=True)
print(f"Saved gene metadata to: {gene_metadata_parquet_path}")

# --- 5. Save adata.obs (Observation/Cell Metadata) as cell_metadata.parquet ---
cell_metadata_parquet_path = os.path.join(OUTPUT_DIR_NAME, "cell_metadata.parquet")
for col in adata.obs.select_dtypes(include=['category']).columns:
    adata.obs[col] = adata.obs[col].astype(str) # Convert categoricals to string for parquet compatibility
adata.obs.index.name = "cell_id"
adata.obs.to_parquet(cell_metadata_parquet_path, index=True)
print(f"Saved cell metadata to: {cell_metadata_parquet_path}")


# --- 6. Perform PCA and save results (or use pre-computed) ---
if perform_pca_flag:
    print(f"\nStarting PCA with {N_PCA_COMPONENTS} components...")
    # Compute Highly Variable Genes if not present and subset adata for DR
    if 'highly_variable' not in adata.var.columns or not adata.var['highly_variable'].any():
        print("  'highly_variable' column not found or no HVGs marked. Computing highly variable genes...")
        # Normalize and log1p might be needed if adata.X are raw counts, otherwise skip if already preprocessed
        # sc.pp.normalize_total(adata, target_sum=1e4) # Uncomment if counts need normalization
        # sc.pp.log1p(adata) # Uncomment if not already log-transformed
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=N_TOP_HVGS, subset=True)
        print(f"  Identified {adata.var['highly_variable'].sum()} highly variable genes.")
    else:
        print(f"  Using pre-computed highly variable genes. Number of HVGs: {adata.var['highly_variable'].sum()}")
        # Ensure adata is subsetted to HVGs for PCA/UMAP
        if not adata.uns.get('hvg_subset_performed', False): # Check if already subsetted by this script
            adata = adata[:, adata.var['highly_variable']].copy()
            adata.uns['hvg_subset_performed'] = True # Mark that subsetting was done
            print(f"  Subsetting adata to HVGs. New shape: {adata.shape}")


    # Ensure the data for PCA is dense
    X_for_pca = adata.X # Use the (potentially subsetted to HVG) expression matrix

    if APPLY_SCALING_BEFORE_PCA:
        print("  Scaling expression data before PCA...")
        scaler = StandardScaler()
        X_for_pca = scaler.fit_transform(X_for_pca)
    else:
        print("  Skipping scaling before PCA.")

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    pca_transformed_data = pca.fit_transform(X_for_pca)

    # Create a DataFrame for the PCA transformed data
    pca_columns = [f"PC{i+1}" for i in range(N_PCA_COMPONENTS)]
    df_pca = pd.DataFrame(pca_transformed_data, index=adata.obs_names, columns=pca_columns)
    df_pca.index.name = "cell_id"

    pca_parquet_path = os.path.join(OUTPUT_DIR_NAME, "pca_embeddings.parquet")
    df_pca.to_parquet(pca_parquet_path, index=True)
    print(f"Saved PCA embeddings to: {pca_parquet_path}")

    # Save the explained variance ratio
    df_explained_variance = pd.DataFrame({
        'PrincipalComponent': [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        'ExplainedVarianceRatio': pca.explained_variance_ratio_,
        'CumulativeExplainedVarianceRatio': np.cumsum(pca.explained_variance_ratio_)
    })
    explained_variance_parquet_path = os.path.join(OUTPUT_DIR_NAME, "pca_explained_variance.parquet")
    df_explained_variance.to_parquet(explained_variance_parquet_path, index=False)
    print(f"Saved PCA explained variance ratio to: {explained_variance_parquet_path}")

else: # If precomputed PCA was found
    print(f"\nUsing pre-computed PCA embeddings from adata.obsm['X_pca'].")
    pca_columns = [f"PC{i+1}" for i in range(precomputed_pca.shape[1])]
    df_pca = pd.DataFrame(precomputed_pca, index=adata.obs_names, columns=pca_columns)
    df_pca.index.name = "cell_id"
    pca_parquet_path = os.path.join(OUTPUT_DIR_NAME, "pca_embeddings.parquet")
    df_pca.to_parquet(pca_parquet_path, index=True)
    print(f"Saved pre-computed PCA embeddings to: {pca_parquet_path}")


# --- 7. Perform UMAP and save results (or use pre-computed) ---
if perform_umap_flag:
    print(f"\nStarting UMAP with {N_UMAP_COMPONENTS} components...")
    # UMAP typically runs on pre-computed PCA components
    X_for_umap = df_pca.values # Use PCA results as input for UMAP

    if X_for_umap.shape[1] == 0:
        print("Warning: X_for_umap is empty (PCA might have failed or returned no components). Skipping UMAP calculation.")
    else:
        reducer = umap.UMAP(n_components=N_UMAP_COMPONENTS,
                            n_neighbors=UMAP_N_NEIGHBORS,
                            min_dist=UMAP_MIN_DIST,
                            random_state=42, # For reproducibility
                            transform_seed=42) # For reproducibility if transform is called later

        umap_embeddings = reducer.fit_transform(X_for_umap)

        umap_columns = [f"UMAP{i+1}" for i in range(N_UMAP_COMPONENTS)]
        df_umap = pd.DataFrame(umap_embeddings, index=adata.obs_names, columns=umap_columns)
        df_umap.index.name = "cell_id"

        umap_parquet_path = os.path.join(OUTPUT_DIR_NAME, "umap_embeddings.parquet")
        df_umap.to_parquet(umap_parquet_path, index=True)
        print(f"Saved UMAP embeddings to: {umap_parquet_path}")
else: # If precomputed UMAP was found
    print(f"\nUsing pre-computed UMAP embeddings from adata.obsm['X_umap'].")
    umap_columns = [f"UMAP{i+1}" for i in range(precomputed_umap.shape[1])]
    df_umap = pd.DataFrame(precomputed_umap, index=adata.obs_names, columns=umap_columns)
    df_umap.index.name = "cell_id"
    umap_parquet_path = os.path.join(OUTPUT_DIR_NAME, "umap_embeddings.parquet")
    df_umap.to_parquet(umap_parquet_path, index=True)
    print(f"Saved pre-computed UMAP embeddings to: {umap_parquet_path}")


# --- 8. Extract and Save Highly Variable Genes (HVGs) metadata ---
if 'highly_variable' in adata.var.columns and adata.var['highly_variable'].any():
    df_hvg = adata.var[adata.var['highly_variable']].copy()
    hvg_metadata_parquet_path = os.path.join(OUTPUT_DIR_NAME, "highly_variable_gene_metadata.parquet")
    df_hvg.index.name = "gene_id"
    df_hvg.to_parquet(hvg_metadata_parquet_path, index=True)
    print(f"Saved highly variable gene metadata to: {hvg_metadata_parquet_path}")
else:
    print("\n'highly_variable' column not found or no HVGs marked in adata.var. Skipping saving HVG metadata.")


# --- 9. Calculate and Save Basic Gene Statistics ---
print("\nCalculating basic gene statistics...")
df_gene_stats = pd.DataFrame(index=adata.var_names)
df_gene_stats.index.name = "gene_id"

df_gene_stats['mean_expression'] = np.asarray(adata.X).mean(axis=0)
df_gene_stats['n_cells_expressed'] = np.asarray(adata.X > 0).sum(axis=0)

gene_stats_parquet_path = os.path.join(OUTPUT_DIR_NAME, "gene_statistics.parquet")
df_gene_stats.to_parquet(gene_stats_parquet_path, index=True)
print(f"Saved gene statistics to: {gene_stats_parquet_path}")


# --- 10. Calculate and Save Cell Type Proportions ---
print("\nCalculating cell type proportions...")

cell_type_col = None
# Prioritize common cell type names
for col_name in ['cell_type', 'cell_identity', 'celltype', 'CellType']:
    if col_name in adata.obs.columns:
        cell_type_col = col_name
        break

if cell_type_col:
    print(f"Using '{cell_type_col}' for cell type proportions.")
    # Overall proportions
    df_cell_type_proportions_overall = adata.obs[cell_type_col].value_counts(normalize=True).reset_index()
    df_cell_type_proportions_overall.columns = [cell_type_col, 'proportion']
    df_cell_type_proportions_overall.to_parquet(os.path.join(OUTPUT_DIR_NAME, "cell_type_proportions_overall.parquet"), index=False)
    print("Saved overall cell type proportions.")

    # Grouped proportions by age, donor, or group
    grouping_column = None
    # Prioritize common age/grouping names for mouse data
    for col in ['age', 'Age', 'age_group', 'AgeGroup', 'donor_id', 'sample_id', 'genotype']:
        if col in adata.obs.columns:
            grouping_column = col
            break

    if grouping_column:
        print(f"Calculating cell type proportions by '{grouping_column}'...")
        df_cell_type_proportions_grouped = adata.obs.groupby(grouping_column)[cell_type_col].value_counts(normalize=True).unstack(fill_value=0)
        df_cell_type_proportions_grouped.index.name = grouping_column
        df_cell_type_proportions_grouped.to_parquet(os.path.join(OUTPUT_DIR_NAME, f"cell_type_proportions_by_{grouping_column}.parquet"), index=True)
        print(f"Saved cell type proportions by '{grouping_column}'.")
    else:
        print("  Could not find a suitable grouping column (e.g., 'age', 'donor_id') in adata.obs for grouped cell type proportions.")
else:
    print("  No suitable cell type column found in adata.obs. Skipping cell type proportion analysis.")


# --- 11. Generate Sample/Donor Metadata ---
print("\nAggregating donor/sample metadata...")
donor_id_col = None
# Prioritize common donor/sample ID names for mouse data
for col in ['donor_id', 'sample_id', 'mouse_id', 'group_id']:
    if col in adata.obs.columns:
        donor_id_col = col
        break

if donor_id_col:
    print(f"Using '{donor_id_col}' for donor/sample metadata aggregation.")
    df_donor_metadata = adata.obs.groupby(donor_id_col).first().reset_index()

    # Define a list of common relevant donor-level metadata columns for mouse studies
    cols_to_keep_for_donor = [
        donor_id_col, 'age', 'Age', 'age_group', 'AgeGroup', 'sex', 'Sex', 'genotype', 'Genotype',
        'tissue', 'Tissue', 'condition', 'Condition', 'batch', 'Batch'
    ]
    # Filter df_donor_metadata to only include columns that were in original obs AND are in cols_to_keep_for_donor
    df_donor_metadata = df_donor_metadata[[col for col in cols_to_keep_for_donor if col in df_donor_metadata.columns]]

    df_donor_metadata.set_index(donor_id_col, inplace=True)
    df_donor_metadata.to_parquet(os.path.join(OUTPUT_DIR_NAME, "donor_metadata.parquet"), index=True)
    print(f"Saved donor/sample metadata to: {os.path.join(OUTPUT_DIR_NAME, 'donor_metadata.parquet')}")
else:
    print("  Could not find a suitable donor/sample ID column (e.g., 'donor_id', 'mouse_id') in adata.obs. Skipping donor metadata aggregation.")


print(f"\nAll relevant Parquet files for Mouse Muscle Aging Atlas have been created in the '{OUTPUT_DIR_NAME}' directory.")
print("You can now use these files for your submission to the hackathon!")
