#!/usr/bin/env Rscript

# Load libraries
library(scry)
library(SingleCellExperiment)
library(anndataR)

args = commandArgs(trailingOnly=TRUE)

# Assign arguments to variable names
in_path <- args[1]
out_path <- args[2]

# Load in data
sce <- read_h5ad(in_path, as = "SingleCellExperiment")

# Feature selection
sce <- devianceFeatureSelection(sce, assay = "X")

# Save data
write_h5ad(sce, out_path)
