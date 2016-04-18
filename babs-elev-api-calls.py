# Keeping this as a separate file to make sure I don't accidentally make a bunch of api calls
from functools import reduce
import sys, os, getopt, pandas as pd, numpy as np, urllib.request, json

base_url = 'https://maps.googleapis.com/maps/api/elevation/json?'
KEY = ''

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

def getElevationFromAPI(station_data, key=KEY):
    elev = []

    for i in np.arange(len(station_data.index.values)):
        print(station_data['lat'][i], station_data['long'][i])
        request = urllib.request.urlopen('{}locations={},{}&key={}'.format(base_url,str(station_data['lat'][i]), str(station_data['long'][i]), key))
        jsonrequest = json.loads(request.read().decode())
        elev.append(jsonrequest['results'][0]['elevation'])

    return elev

# Pass Google Elevation API Key as -k argument
def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'hk:', ['apikey='])
        for opt, arg in opts:
            if opt == '-h':
                print('babs-elev-api-calls.py -k <google_elevation_api_key>')
                sys.exit()
            elif opt in ('-k', '--apikey'):
                try:
                    KEY = str(arg)
                    print("Running with api key: ", KEY)
                except:
                    print('Please include a api key.')
                    sys.exit()
    except getopt.GetoptError:
        print('babs-elev-api-calls.py -k <google_elevation_api_key>')

    station_data = LoadData(os.path.join(os.path.dirname(__file__),'generated_files'), date_format='%m/%d/%Y', date_headers=['installation'])




if __name__ == "__main__":
    main(sys.argv[1:])
