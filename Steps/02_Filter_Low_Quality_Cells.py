#!/usr/bin/env python3
import argparse
from pathlib import Path
import logging
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation

## Custom outlier function 
def is_outlier(adata, metric: str, nmads: int):
	M = adata.obs[metric]
	outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
		np.median(M) + nmads * median_abs_deviation(M) < M
	)
	return outlier
	
def main(args):
	outdir = Path(args.output_dir)
	outdir.mkdir(parents=True, exist_ok=True)
	log = outdir / f"{args.sample}_step02.log"
	logging.basicConfig(level=logging.INFO, handlers=[
		logging.FileHandler(str(log)),
		logging.StreamHandler()
	], format="%(asctime)s %(levelname)s: %(message)s")
	logging.info("Starting step 2: filter low quality cells")

	# Load data
	input_path = Path(args.input)
	if not input_path.exists():
		logging.error(f"Input path not found: {input_path}")
		raise SystemExit(1)

	adata = sc.read_h5ad(input_path)
	adata.raw = adata.copy()

	logging.info(adata)

	# Filter low quality cells
	adata.obs["outlier"] = (
		is_outlier(adata, "log1p_total_counts", 5)
		| is_outlier(adata, "log1p_n_genes_by_counts", 5)
		| is_outlier(adata, "pct_counts_in_top_20_genes", 5)
	)
	logging.info(f"Outliers by library/gene/top20: {adata.obs['outlier'].sum()}")

	adata.obs["MT_outlier"] = is_outlier(adata, "pct_counts_MT", 3) | (adata.obs["pct_counts_MT"] > 8)
	logging.info(f"MT outliers: {adata.obs['MT_outlier'].sum()}")
	
	keep = (~adata.obs["outlier"]) & (~adata.obs["MT_outlier"])
	logging.info("Keeping %d / %d cells", keep.sum(), adata.n_obs)
	adata = adata[keep].copy()

	logging.info(adata)
	
	# Write filtered output (Keeping your file saving step exactly as requested)
	out_h5 = outdir / f"{args.sample}_qc_filtered.h5ad"
	adata.write(out_h5)
	logging.info(f"Wrote filtered AnnData: {out_h5}")
	
	# Save post-filter-soupx histogram (Using the newly calculated log1p column)
	p2 = sns.displot(adata.obs["log1p_total_counts"], bins=100, kde=False)
	p2.set_xlabels("log1p(SoupX Corrected Total Counts)")
	post_hist = outdir / f"{args.sample}_post_filter_soup_library_depth_histogram.png"
	plt.savefig(post_hist)
	plt.close()
	logging.info(f"Wrote post-filter and soupx histogram: {post_hist}")

	# Save post-filter scatter
	post_scatter = outdir / f"{args.sample}_post_filter_scatter.png"
	sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_MT", show=False)
	plt.savefig(post_scatter)
	plt.close()
	logging.info(f"Wrote post-filter scatter: {post_scatter}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Filter low-quality single-cell barcodes")
	parser.add_argument("--input", required=True, help="Path to ambient_removed.h5ad")
	parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
	parser.add_argument("--sample", required=True, help="Sample name (used for output filenames)")
	parser.add_argument("--threads", type=int, default=1, help="Number of threads (optional)")
	args = parser.parse_args()
	main(args)
