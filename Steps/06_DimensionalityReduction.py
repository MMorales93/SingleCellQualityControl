#!/usr/bin/env python3

import scanpy as sc
import argparse
import logging
import matplotlib.pyplot as plt
from pathlib import Path

def main(args): 
	outdir = Path(args.output_dir)
	outdir.mkdir(parents=True, exist_ok=True)
	
	log = outdir / f"{args.output_prefix}_step06_DimensionalityReduction.log"
	logging.basicConfig(level=logging.INFO, handlers=[
		logging.FileHandler(str(log)),
		logging.StreamHandler()
	], format="%(asctime)s %(levelname)s: %(message)s")
	
	logging.info("Started step 6: Dimensionality Reduction")
	
	# Import data
	input_path = Path(args.input)
	if not input_path.exists():
		logging.error(f"Input path not found: {input_path}")
		raise SystemExit(1)
		
	adata = sc.read_h5ad(str(input_path))
	logging.info(f"Read in data from step 5: {input_path}")
	
	## PCA ##
	logging.info(f"Starting PCA on log1p_norm layer using highly_deviant genes...")
	sc.pp.pca(adata, svd_solver="arpack", layer="log1p_norm", mask_var='highly_deviant')
	
	# Plot
	pca_plot = outdir / f"{args.output_prefix}_PCA_plot.png"
	sc.pl.pca_scatter(adata, color='total_counts', show=False)
	plt.savefig(pca_plot, dpi=300, bbox_inches="tight")
	plt.close()
	logging.info(f"Wrote PCA diminsion reduction plot: {pca_plot}")
	
	
	## t-SNE ##
	logging.info(f"Starting t-SNE...")
	sc.tl.tsne(adata, use_rep="X_pca")
	
	# Plot
	tsne_plot = outdir / f"{args.output_prefix}_tSNE_plot.png"
	sc.pl.tsne(adata, color='total_counts', show=False)
	plt.savefig(tsne_plot, dpi=300, bbox_inches="tight")
	plt.close()
	logging.info(f"Wrote t-SNE diminsion reduction plot: {tsne_plot}")
	
	## UMAP ##
	logging.info(f"Starting UMAP...")
	# Neighborhood graph
	sc.pp.neighbors(adata, use_rep="X_pca")
	sc.tl.umap(adata)
	
	# Plot
	umap_plot = outdir / f"{args.output_prefix}_UMAP_plot.png"
	sc.pl.umap(adata, color=['total_counts', 'pct_counts_MT', 'pct_counts_RIBO', 'pct_counts_HB','scrublet_score', 'scrublet_doublet', 'doubletdetection_score', 'doubletdetection_doublet'])
	plt.savefig(umap_plot)
	plt.close()
	logging.info(f"Wrote UMAP diminsion reduction plots: {umap_plot}")
	
	# Save h5ad file 
	out_h5 = outdir / f"{args.output_prefix}_dim_reduced.h5ad"
	adata.write(out_h5)
	logging.info(f"Wrote dimensionality reduced AnnData: {out_h5}")
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Dimensionality Reduction with scanpy")
	parser.add_argument("--input", required = True, help = "Path to h5ad file output from step 6: Feature Selection.")
	parser.add_argument("--output-prefix", required=True, help="Prefix for output files/logs.")
	parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
	args = parser.parse_args()
	main(args)