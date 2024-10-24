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
            start = int(line[1])
            stop = int(line[2])
            if start > stop:
                start = int(line[2])
                stop = int(line[1])
            if len(line) < 3:
                continue
            if 'Y' in line[0] or 'X' in line[0] or 'M' in line[0]:
                continue
            if line[0] in list(chroms.keys()) and stop < chroms[line[0]]:
                regions.append([line[0],start,stop])
    f.close()
    return regions

def prepareDataBed(inputBW,regions):
    inputData = []
    # Iterate over each common chromosome
    with pyBigWig.open(inputBW) as bw_input:
        for chrom,start,stop in regions:
            input_values = np.nan_to_num(np.array(bw_input.values(chrom, start,stop)))
            if np.sum(input_values) <= 0:
                inputData.append(0)
            else:
                inputData.append(np.sum(np.nan_to_num(input_values))/((stop-start)/1000))
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

def randomForestRegressor(inputDF,numEst):
    features = [(i) for i in inputDF.columns.to_list() if i != 'Target' ]
    X = inputDF.loc[:,features]
    y = inputDF.Target
    #y_trans = quantile_transform(y.to_frame(), n_quantiles=1000, output_distribution="normal", copy=True).squeeze()
    #X_trans = quantile_transform(y.to_frame(), n_quantiles=1000, output_distribution="normal", copy=True).squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    #norm = MinMaxScaler().fit(X_train)
    #X_train_norm = norm.transform(X_train)
    #X_test_norm = norm.transform(X_test)
    model = RandomForestRegressor(n_estimators=numEst,random_state=42,min_samples_split=10, min_samples_leaf=5)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    precision = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions) 
    #accuracy = accuracy_score(y_test, predictions)
    result = permutation_importance(model, X_train, y_train, n_repeats=10,
                                random_state=42,n_jobs=10) 
    return model , precision, mse,result ,features 


def find_best_predictor(input_file, numEst):
    """
    Find the best predicting bigWig file based on mean squared error.
    """
    inputDF = pd.read_pickle(input_file)
    start = time.perf_counter()
    inputDF.dropna(inplace=True)
    inputDF= inputDF.loc[~(inputDF==0).all(axis=1)]
    quantiles = int(len(inputDF.index))
    for i in inputDF.columns.to_list():
        inputDF[i]= quantile_transform(inputDF[i].to_frame(), n_quantiles=quantiles,output_distribution="normal", copy=True,subsample=quantiles).squeeze()
    print('Starting Modeling',flush=True)
    sys.stdout.flush()
    start = time.perf_counter()
    model , precision, mse,weights, features  = randomForestRegressor(inputDF,numEst)
    end = time.perf_counter()
    print(f'Modeling Time:'+str(end-start),flush=True)
    sys.stdout.flush()
    print('MSE: '+str(mse))
    print('R2: '+str(precision))
    #print('Accuracy: '+str(accuracy))
    print(f'Name\tImport\tImportMean\tStD')
    for i,j, k,l in zip(features,model.feature_importances_,weights.importances_mean,weights.importances_std):
        print(i+':\t'+str(j)+'\t'+str(k)+'\t'+str(l))
    pd.set_option('display.max_colwidth', None)
    print('Spearman Rho:')
    print(inputDF.corrwith(inputDF['Target'],method='spearman'))
    return  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine the best predicting bigWig file.")
    parser.add_argument('-i',"--inPickle", help="Input pickle file")
    parser.add_argument("-n","--numEstimator",type=int, default=10000, help="Number of estimators (default: 10000)")
    #parser.add_argument("-p","--#",type=int, default=2, help="Number of Processors (default: 2)")
    args = parser.parse_args()

    input_file = args.inPickle
    numEst = args.numEstimator
    find_best_predictor(input_file, numEst)

 
