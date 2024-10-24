import argparse
import sys
import time
import pyBigWig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, roc_auc_score, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import quantile_transform
def prepare_data(input_file, regions):
    """
    Prepare data for machine learning by binning values from bigWig files or using regions specified in a BED file.
    """

    # Get chromosome names and lengths from both the input and target bigWig files

    inputData = []
    # Iterate over each common chromosome
    with pyBigWig.open(input_file) as bw_input:
        for chrom,num_bin,length in regions:
            input_values = np.array(bw_input.stats(chrom, 0, length, nBins=num_bin,numpy=True,exact=True))
            inputData.append(input_values)
    inputData = np.concatenate(inputData,axis=0)
    return inputData

def bed2Regions(bedFile,chroms):
    regions = []
    with open(bedFile, 'r') as f:
        for line in f: 
            line = line.strip().split('\t')
            if len(line) < 3:
                continue
            start = int(line[1])
            stop = int(line[2])
            if start > stop:
                start = int(line[2])
                stop = int(line[1])
            if 'Y' in line[0] or 'M' in line[0]:
                continue
            if line[0] in list(chroms.keys()) and stop < chroms[line[0]]:
                regions.append([line[0],start,stop])
    f.close()
    return regions

def prepareDataBedv2(inputBW,regions):
    inputData = []
    # Iterate over each common chromosome
    with pyBigWig.open(inputBW) as bw_input:
        for chrom,start,stop in regions:
            if start > stop:
                input_values = np.mean(np.nan_to_num(np.array(bw_input.values(chrom, stop,start))))
            else:
                input_values = np.mean(np.nan_to_num(np.array(bw_input.values(chrom, start,stop))))
            if input_values == 0:
                inputData.append(0)
            else:
                inputData.append(input_values)
    return inputData

def prepareDataBed(inputBW,regions):
    inputData = []
    # Iterate over each common chromosome
    with pyBigWig.open(inputBW) as bw_input:
        for chrom,start,stop in regions:
            if start > stop:
                input_values = np.max(np.nan_to_num(np.array(bw_input.values(chrom, stop,start))))
            else:
                input_values = np.max(np.nan_to_num(np.array(bw_input.values(chrom, start,stop))))
            inputData.append(input_values)
    return inputData

def prepareDataBedv3(inputBW,regions):
    inputData = []
    # Iterate over each common chromosome
    with pyBigWig.open(inputBW) as bw_input:
        for chrom,start,stop in regions:
            if start > stop:
                input_values = np.sum(np.nan_to_num(np.array(bw_input.values(chrom, stop,start))))
            else:
                input_values = np.sum(np.nan_to_num(np.array(bw_input.values(chrom, start,stop))))
            inputData.append(input_values)
    return inputData



def prepDatabins(inputBW,regions):
    inputData = []
    # Iterate over each common chromosome
    with pyBigWig.open(inputBW) as bw_input:
        for chrom,start,stop in regions:
            inputData.append(np.sum(np.nan_to_num(np.array(bw_input.values(chrom, start,stop)))))

    return inputData

def generate_bins(chrom_sizes, bin_size=5000):
    bins = []
    for chrom, size in chrom_sizes:
        for start in range(0, size, bin_size):
            end = min(start + bin_size, size)
            bins.append((chrom, start, end))
    return bins


def data2pickle(input_files, target_file, bin_size,bed_file=None,outname=None):
    """
    Find the best predicting bigWig file based on mean squared error.
    """
    inputDF = pd.DataFrame()
    inputDFv2 = pd.DataFrame()
    inputDFv3 = pd.DataFrame()
    start = time.perf_counter()
    chroms = {}
    common_chromosomes = set(pyBigWig.open(input_files[0]).chroms()) & set(pyBigWig.open(target_file).chroms())
    chromosomesSizes = pyBigWig.open(input_files[0]).chroms()
    for i in common_chromosomes: 
        if 'Y' in i or 'X' in i or 'M' in i:
            continue 
        chroms[i] = chromosomesSizes[i]
    if bed_file != None:
        regions = bed2Regions(bed_file,chroms)
        for file in input_files:
            prepData = prepareDataBed(file, regions)
            inputDF[file] = prepData
            inputDFv2[file] = prepareDataBedv2(file, regions)
            inputDFv3[file] = prepareDataBedv3(file, regions)
        inputDF['Target'] = prepareDataBed(target_file, regions)
        inputDFv2['Target'] = prepareDataBedv2(target_file, regions)
        inputDFv3['Target'] = prepareDataBedv3(target_file, regions)
    inputDF.dropna(inplace=True)
    inputDF.to_pickle(outname+'.Max.pickle')
    inputDFv2.dropna(inplace=True)
    inputDFv2.to_pickle(outname+'.Mean.pickle')
    inputDFv3.dropna(inplace=True)
    inputDFv3.to_pickle(outname+'.Sum.pickle')

    return  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine the best predicting bigWig file.")
    parser.add_argument('-i',"--inBW", nargs="+", help="List of input bigWig files")
    parser.add_argument('-t',"--tBW", help="Target bigWig file")
    parser.add_argument("--bs", type=int, default=5000, help="Genomic bin size for binning values (default: 5000)")
    parser.add_argument("--bf", type=str, default=None, help="Use bedfile for the regions")
    parser.add_argument("--outname", type=str, default=None, help="Pickle prefix for saving data file")
    args = parser.parse_args()

    input_files = args.inBW
    target_file = args.tBW
    bs = args.bs
    bed_file = args.bf
    outname = args.outname
    data2pickle(input_files, target_file, bs,bed_file,outname)

 
