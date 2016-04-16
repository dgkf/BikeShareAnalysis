# Python 3.4.4
from functools import reduce
import sys, os, re, math, datetime, numpy as np, pandas as pd

# radius of earth in kilometers
EARTHR = 6371
# conversion factor for going from degrees to km
degToKm = 2 * math.pi * EARTHR / 360

def LoadData(data_dir, chunksize=30000, req_headers=[], date_headers=[], date_format='', verbose=False, repstrs=(), nrows=None):
    # get file names
    data_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    # loop through files and make a list of DataFrames
    data = []
    for f in data_files:
        if verbose: print('reading data file ' + str(f))

        # get entire list of headers if no required headers are specified
        # moved header handling to per-file to deal with multiple files with slightly different headers
        header = pd.read_csv(f, nrows=0).columns.values
        header_final = [reduce(lambda a, kv: a.replace(*kv), repstrs, i) for i in header]
        req_h = header if not req_headers else req_headers
        cols = [h for h in req_h if reduce(lambda a, kv: a.replace(*kv), repstrs, h) in header_final]
        datecols = date_headers
        datecols = [h for h in req_h if reduce(lambda a, kv: a.replace(*kv), repstrs, h) in datecols]

        # build read_csv arguments
        farg = {'delimiter': ',', 'usecols': cols}
        if nrows is None:
            farg['chunksize'] = chunksize
        else:
            farg['nrows'] = nrows

        # Read files from csv
        if 'chunksize' in farg:
            chunkreader = pd.read_csv(f, **farg)
            for i, chunk in enumerate(chunkreader):
                if verbose: print('reading chunk: ', i+1, '( lines:', i*chunksize, 'to', (i+1)*chunksize, ')')
                # Parse datetimes
                for d in datecols:
                    chunk[d] =  pd.to_datetime(chunk[d],format=date_format)

                chunk.columns = [reduce(lambda a, kv: a.replace(*kv), repstrs, i) for i in chunk.columns.values]
                data.append(chunk)

            # return single concatenated DataFrame
            return pd.concat(data, ignore_index=True)
        else:
            return pd.read_csv(f, **farg)

def LoadGeneratedStationData(data_dir, chunksize=30000, req_headers=[], verbose=False):
    if verbose: print('loading data.')
    date_format = '%m/%d/%Y'

    return LoadData(data_dir, chunksize=chunksize, date_format=date_format)

def LoadBABSWeatherData(data_dir, chunksize=30000, req_headers=[], verbose=False):
    if verbose: print('loading data.')

    headers = pd.read_csv()

    # build our file reading inputs

    date_format = '%m/%d/%Y'

    return LoadData(data_dir, cols, datecols, req_headers=req_headers, chunksize=chunksize, date_format=date_format)

def LoadBABSStatusData(data_dir, chunksize=30000, req_headers=[], verbose=False):
    if verbose: print('loading data.')
    date_format = '%m/%d/%Y %H:%M'

    return LoadData(data_dir, chunksize=chunksize, req_headers=req_headers, date_format=date_format)

def LoadBABSTripData(data_dir, chunksize=30000, req_headers=['Trip ID','Duration','Start Date','Start Station','Start Terminal','End Date','End Station','End Terminal','Bike \#','Subscription Type','Zip Code'], verbose=False):
    if verbose: print('loading data.')

    # build our file reading inputs
    cols = ['Trip ID','Duration','Start Date','Start Station','Start Terminal','End Date','End Station','End Terminal','Bike \#','Subscription Type','Zip Code']
    cols = [h for h in req_headers if h in cols]
    datecols = ['Start Date', 'End Date']
    datecols = [h for h in req_headers if h in datecols]

    date_format = '%m/%d/%Y %H:%M'

    return LoadData(data_dir, cols, datecols, chunksize=chunksize, date_format=date_format)

def main(argv):
    # station_data = LoadData(os.path.join(os.path.dirname(__file__),'generated_files'), date_format='%m/%d/%Y', date_headers=['installation'])
    #
    # weather_data = LoadData(os.path.join(os.path.dirname(__file__),'babs_archives','weather_data'), date_headers=['Date'], date_format='%m/%d/%Y', repstrs=((' ', ''), ('_', ''), ('PDT', 'Date'), ('zip', 'Zip'), ('In', '')))
    #
    # trip_data = LoadData(os.path.join(os.path.dirname(__file__),'babs_archives','trip_data'), date_headers=['Start Date', 'End Date'], date_format='%m/%d/%Y %H:%M', verbose=True, nrows=30, req_headers=['Start Date','Start Terminal','End Date','End Terminal','Subscription Type'])
    return 1

if __name__ == "__main__":
    main(sys.argv[1:])
