#!/usr/bin/env python3

import scanpy as sc
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def main(args):
	outdir = Path(args.output_dir)
	outdir.mkdir(parents=True, exist_ok=True)
	
	log = outdir / f"{args.output_prefix}_step07_Clustering.log"
	logging.basicConfig(level=logging.INFO, handlers=[
		logging.FileHandler(str(log)),
		logging.StreamHandler()
	], format="%(asctime)s %(levelname)s: %(message)s")
	
	logging.info("Started step 7: Clustering")
	
	# Import data
	input_path = Path(args.input)
	if not input_path.exists():
		logging.error(f"Input path not found: {input_path}")
		raise SystemExit(1)
		
	adata = sc.read_h5ad(str(input_path))
	logging.info(f"Read in data from Step 6: {input_path}")
	
	# Clustering
	sc.pp.neighbors(adata, n_pcs=30)
	sc.tl.umap(adata)
	
	resolutions = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
	
	for resolution in resolutions:
	    sc.tl.leiden(adata, resolution = resolution, flavor="igraph", n_iterations=2, key_added=f"leiden_res{resolution}")
	
	#sc.tl.leiden(adata, key_added="leiden_res0.25", resolution=0.25, flavor="igraph", n_iterations=2)
	
	# Generate Plots
	umap_plot = outdir / f"{args.output_prefix}_ClusteringUMAP_Group1.png"
	
	res_keys = [f"leiden_res{r}" for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
	
	sc.pl.umap(adata, color=res_keys, legend_loc="on data", show=False)
	plt.savefig(umap_plot)
	plt.close()
	logging.info(f"Wrote Clustering UMAP plots: {umap_plot}")
	
	#Save h5ad file
	out_h5 = outdir / f"{args.output_prefix}_clustered.h5ad"
	adata.write(out_h5)
	logging.info(f"Wrote Clustered AnnData: {out_h5}")
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Clustering with scanpy")
	parser.add_argument("--input", required = True, help = "Path to h5ad file output from step 6: Dimensionality Reduction")
	parser.add_argument("--output-prefix", required = True, help = "Prefix for output files and logs")
	parser.add_argument("--output-dir", required = True, help = "Directory to write outputs")
	args = parser.parse_args()
	main(args)