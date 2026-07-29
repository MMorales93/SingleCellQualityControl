#!/usr/bin/env python3
import scanpy as sc
import soupx
import pandas as pd
import argparse
from anndata import AnnData
from pathlib import Path
import logging
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import seaborn as sns

def main(args):
	outdir = Path(args.output_dir)
	outdir.mkdir(parents=True, exist_ok=True)
	log = outdir / f"{args.sample}_step01.log"
	logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(str(log)),
        logging.StreamHandler()
	], format="%(asctime)s %(levelname)s: %(message)s")
	logging.info("Starting step 1: remove ambient RNA")

	# Read Data
	adata_raw = sc.read_10x_mtx(str(args.raw_10x), cache=True)
	adata_raw.var_names_make_unique()
	adata_raw.raw = adata_raw.copy()

	adata_filt = sc.read_10x_mtx(str(args.input), cache=True)
	adata_filt.var_names_make_unique()
	
	logging.info("adata_raw shape: %s", adata_raw.shape)
	logging.info("adata_filt shape: %s", adata_filt.shape)

	# Annotate QC Genes
	adata_raw.var["MT"] = adata_raw.var_names.str.startswith("MT-")
	# ribosomal genes
	adata_raw.var["RIBO"] = adata_raw.var_names.str.startswith(("RPS", "RPL"))
	# hemoglobin genes
	adata_raw.var["HB"] = adata_raw.var_names.str.contains(r"^HB[ABDEGMQZ]\d*(?!\w)")

	# Calculate QC metrics
	sc.pp.calculate_qc_metrics(adata_raw, qc_vars=["MT", "RIBO", "HB"], inplace=True, percent_top=[20], log1p=True)
	logging.info("Calculated QC metrics")

	# Make pre-filter plots
	pre_hist = outdir / f"{args.sample}_pre_QC_library_depth_histogram.png"
	sns.displot(adata_raw.obs["log1p_total_counts"], bins=100, kde = False)
	plt.savefig(pre_hist)
	plt.close()
	logging.info(f"Wrote pre-filter histogram: {pre_hist}")

	pre_violin_mt = outdir / f"{args.sample}_pre_QC_MT_violin.png"
	sc.pl.violin(adata_raw, "pct_counts_MT", show=False)
	plt.savefig(pre_violin_mt)
	plt.close()
	logging.info(f"Wrote pre-filter MT violin: {pre_violin_mt}")
	
	pre_violin_ribo = outdir / f"{args.sample}_pre_QC_RIBO_violin.png"
	sc.pl.violin(adata_raw, "pct_counts_RIBO", show=False)
	plt.savefig(pre_violin_ribo)
	plt.close()
	logging.info(f"Wrote pre-filter Ribo violin: {pre_violin_ribo}")
	
	pre_violin_hb = outdir / f"{args.sample}_pre_QC_HB_violin.png"
	sc.pl.violin(adata_raw, "pct_counts_HB", show=False)
	plt.savefig(pre_violin_hb)
	plt.close()
	logging.info(f"Wrote pre-filter HB violin: {pre_violin_hb}")

	pre_scatter = outdir / f"{args.sample}_pre_QC_scatter.png"
	sc.pl.scatter(adata_raw, "total_counts", "n_genes_by_counts", color="pct_counts_MT", show=False)
	plt.savefig(pre_scatter)
	plt.close()
	logging.info(f"Wrote pre-filter scatter: {pre_scatter}")

	logging.info("Raw matrix shape: %s", adata_raw.shape)
	logging.info("Filtered matrix shape: %s", adata_filt.shape)

	# Create SoupChannel
	soup_channel = soupx.SoupChannel(
    tod=adata_raw.X.T.tocsr(),    # raw counts (genes × droplets)
    toc=adata_filt.X.T.tocsr(), # filtered counts (genes × cells)
    metaData=pd.DataFrame(index=adata_filt.obs_names)
	)

	sc.pp.normalize_total(adata_filt, target_sum=1e4)
	sc.pp.log1p(adata_filt)
	sc.pp.pca(adata_filt, svd_solver="arpack")
	sc.pp.neighbors(adata_filt)

	# Add clustering information (essential for good results)
	sc.tl.leiden(adata_filt, resolution=0.5, flavor="igraph")
	soup_channel.setClusters(adata_filt.obs['leiden'].values)

	# Estimate and remove contamination
	soup_channel = soupx.autoEstCont(soup_channel, verbose=True)
	corrected_matrix = soupx.adjustCounts(soup_channel)

	# Create a fresh, completely un-normalized AnnData object
	adata_corrected = sc.AnnData(X=corrected_matrix.T.tocsr())
	adata_corrected.obs_names = adata_filt.obs_names
	adata_corrected.var_names = adata_filt.var_names

	# Re-annotate the QC gene groups on the clean, raw SoupX counts
	adata_corrected.var["MT"] = adata_corrected.var_names.str.startswith("MT-")
	adata_corrected.var["RIBO"] = adata_corrected.var_names.str.startswith(("RPS", "RPL"))
	adata_corrected.var["HB"] = adata_corrected.var_names.str.contains(r"^HB[ABDEGMQZ]\d*(?!\w)")

	# Calculate standard QC metrics on the clean raw corrected counts
	sc.pp.calculate_qc_metrics(
		adata_corrected, 
		qc_vars=["MT", "RIBO", "HB"], 
		inplace=True, 
		percent_top=[20], 
		log1p=True
	)

	# Write output files
	out_h5 = outdir / f"{args.sample}_ambient_removed.h5ad"
	logging.info("corrected_matrix shape: %s", getattr(corrected_matrix, "shape", "unknown"))
	adata_corrected.write(out_h5)
	logging.info(f"Wrote RNA cleaned AnnData: {out_h5}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Remove ambient RNA with SoupX")
	parser.add_argument("--input", required = True, help = "Path to step 1 filtered_feature_bc_matrix directory")
	parser.add_argument("--raw-10x", required = True, help = "Path to raw_feature_bc_matrix directory")
	parser.add_argument("--output-dir", required = True, help = "Directory to write outputs")
	parser.add_argument("--sample", required = True, help = "Sample name")
	parser.add_argument("--threads", type = int, required = True, help = "Threads")
	args = parser.parse_args()
	main(args)