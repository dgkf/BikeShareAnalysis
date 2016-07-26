# Python 3.4
from functools import reduce
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d as art3d
# from mpl_toolkits.basemap import Basemap (I was having issues installing 3.3 alongside 3.4 to use this)
import sys, os, re, math, datetime, random, numpy as np, pandas as pd, urllib.request, json, getopt
from matplotlib import cm, ticker
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

# API Key
KEY = ''

# radius of earth in kilometers
EARTHR = 6371
# conversion factor for going from degrees to km
degToKm = 2 * math.pi * EARTHR / 360
# most recent datapoint
date_most_recent = '20140901'

# disallow api calls as a safegaurd while working
allow_api_calls = False

# asset files and data (for now just going for function)
sf_map_src = 'plot_assets\sf_topo_mikeernst.jpg'
map_y_max = 37.816809
map_y_min = 37.7040638
map_x_min = -122.523110
map_x_max = -122.351960

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

def quadBrezPoints(P0, P1, P2, nSamples):
    ans = np.zeros((nSamples,2))
    for i in range(nSamples):
        t = (i+0.0)/(nSamples-1)
        ans[i,0] = (1-t)**2*P0[0] + 2*(1-t)*t*P1[0] + t**2*P2[0]
        ans[i,1] = (1-t)**2*P0[1] + 2*(1-t)*t*P1[1] + t**2*P2[1]
    return ans

# Made minor edits in photoshop to add figure caption and alter title text & color bar scaling - none of the figure was altered
def plotStations(station, trip):
    sf_stations = list(station[station['landmark']=='San Francisco']['station_id'])
    trip = trip[trip['Start Terminal'].isin(sf_stations) & trip['End Terminal'].isin(sf_stations)] # make sure we're in SF
    trip = trip[~(trip['Start Terminal'] == trip['End Terminal'])] # get rid of entries where someone returns to the same station
    trip = trip.groupby(['Start Terminal', 'End Terminal'])['End Terminal'].count()
    trip = trip.unstack('Start Terminal')

    fig = plt.figure(dpi=72, figsize=(14,14), facecolor='w', edgecolor='w')
    img = mpimg.imread(os.path.join(os.path.dirname(__file__),sf_map_src))
    plt.imshow(img, extent=(map_x_min, map_x_max, map_y_min, map_y_max))

    x = station[station['landmark']=='San Francisco']['long']
    y = station[station['landmark']=='San Francisco']['lat']
    c = station[station['landmark']=='San Francisco']['elev']

    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, c=c, cmap=cm.seismic, s=250, zorder=2, linewidth=3, edgecolor=(1, 1, 1, 0.5))

    max_alt_chng = max(c)-min(c)
    for st in trip.columns.values:
        for et in trip[st].index.values:
            if st != et and not np.isnan(trip[st][et]):
                # draw a fraction of the trips to represent traffic
                for i in range(int(trip[st][et] * 0.2)):

                    bezspread = 0.2 * (random.random() * 0.5 + 0.5)
                    altchng = station[station['station_id']==et]['elev'].values[0]-station[station['station_id']==st]['elev'].values[0]
                    col = (altchng/max_alt_chng if altchng > 0 else 0, 0, -altchng/max_alt_chng if altchng <= 0 else 0, 0.005 + abs(altchng/max_alt_chng) * 0.02)

                    P0 = [station[station['station_id']==st]['long'].values[0], station[station['station_id']==st]['lat'].values[0]]
                    P1 = [station[station['station_id']==et]['long'].values[0], station[station['station_id']==et]['lat'].values[0]]
                    P2 = [(P1[0] + P0[0])*0.5, (P1[1] + P0[1])*0.5]
                    P2[0] += (-1 if altchng > 0 else  1) * (P1[1]-P0[1]) * bezspread
                    P2[1] += ( 1 if altchng > 0 else -1) * (P1[0]-P0[0]) * bezspread

                    PL = quadBrezPoints(P0, P2, P1, 7);

                    plt.plot(PL[:,0], PL[:,1], color=tuple(col), linewidth=1, zorder=1)

    # would love to show a relief map, but py3.3 is not being my friend today, will settle for a static image in the interest of time
    # set resolution=None to skip processing of boundary datasets.
    # m = Basemap(width=12000000,height=9000000,projection='lcc',
    #             resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
    # m.shadedrelief()

    plt.title('Bay Area Bike Share Traffic', fontsize=20)
    plt.xlim((-122.423, -122.386))
    plt.ylim((37.7667, 37.8078))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    plt.tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off', labelbottom='off', labelleft='off', labelright='off')
    cb = plt.colorbar(sc)
    cb.set_label('Elevation (m)')
    fig.set_size_inches(12, 12)
    fig.savefig(os.path.join(os.path.dirname(__file__),'plots','BikeShareTraffic.png'), transparent=True)
    plt.show()

def plotTransitPerCapita(station, trip, lmstats):
    lm_stations = station.groupby(['landmark', 'station_id'])['station_id'].count()
    lm_stations = { lm: list(lm_stations[lm].index.values) for lm in station['landmark'].values}

    # Prep all of the fields we want
    term_st_counts = trip.groupby(['Start Terminal'])['Start Terminal'].count()
    term_st_counts.name = 'Start Terminal Count'
    term_sub_perc_counts = trip.groupby(['Start Terminal', 'Subscription Type'])['Subscription Type'].count()
    term_sub_perc_counts = term_sub_perc_counts.unstack('Subscription Type')
    term_sub_perc_counts = term_sub_perc_counts['Subscriber']
    term_sub_perc_counts.name = 'Subscriber Count'
    term_dur_cum = trip.groupby(['Start Terminal'])['Duration'].sum()
    term_dur_cum.name = 'Cum Duration'
    term_lm = pd.Series([ station[station['station_id'] == i]['landmark'].values[0] for i in term_st_counts.index.values ], index=term_st_counts.index.values)
    term_lm.name = 'landmark'

    # Group by landmark and clean up data
    term_stats = pd.concat([term_lm, term_st_counts, term_sub_perc_counts, term_dur_cum], axis=1)
    term_stats = term_stats.groupby('landmark')['Start Terminal Count', 'Subscriber Count', 'Cum Duration'].sum()
    term_stats['Subscriber Fraction'] = term_stats['Subscriber Count'] / term_stats['Start Terminal Count']
    term_stats['Avg Duration'] = term_stats['Cum Duration'] / term_stats['Start Terminal Count']

    # get some of that altitude data in here!
    lmelev = station.groupby(['landmark'])['elev'].agg({'Elevation Mean': np.mean, 'Elevation Std': np.std})

    # Lump in our population and income stats from wikipedia
    term_stats = pd.concat([term_stats, lmstats, lmelev], axis=1);
    term_stats['household_median_income'] = term_stats['household_median_income'] / 1000
    term_stats['per_cap_income'] = term_stats['per_cap_income'] / 1000
    term_stats['Avg Duration'] = term_stats['Avg Duration'] / 60

    print(term_stats)

    fig = plt.figure()
    fig.suptitle('Trends Across BABS Locations')
    lm_colors = [cm.viridis(x) for x in np.arange(0,5)/4]

    ax = fig.add_subplot(221)
    ind = np.arange(0,5)
    bar_width = 0.6
    bars = ax.bar(ind+bar_width*0.5, term_stats['Start Terminal Count']/term_stats['population'], width=bar_width, color=lm_colors)
    ax.set_xlim(0, 5)
    ax.set_ylabel('Trips Per Capita')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.tick_params(axis='x', which='both', bottom='off', top='off', right='off', left='on', labelbottom='off', labelleft='on', labelright='off')

    ax = fig.add_subplot(223)
    ax.yaxis.grid(b=True, which='major', color='lightgray', linewidth=1.0, linestyle='-')
    ax.scatter(term_stats['per_cap_income'], term_stats['Avg Duration'], c=lm_colors, s=200)
    ax.set_xlabel('Per Capita Income ($1000s)')
    ax.set_ylabel('Avg Trip Duration (min)')
    ax.set_axisbelow(True)

    ax = fig.add_subplot(222)
    ax.yaxis.grid(b=True, which='major', color='lightgray', linewidth=1.0, linestyle='-')
    ax.scatter(term_stats['household_median_income'], term_stats['Subscriber Fraction'], c=lm_colors, s=200)
    ax.set_xlabel('Household Median Income ($1000s)')
    ax.set_ylabel('Subscription Fraction')
    ax.set_axisbelow(True)

    ax = fig.add_subplot(224)
    ind = np.arange(0,5)
    bar_width = 0.6
    bars = ax.bar(ind+bar_width*0.5, term_stats['Elevation Std'], width=bar_width, color=lm_colors)
    ax.set_xlim(0, 5)
    ax.set_ylabel('Station Elevation Std. (m)')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.tick_params(axis='x', which='both', bottom='off', top='off', right='off', left='on', labelbottom='off', labelleft='on', labelright='off')

    fig.legend([mpatches.Patch(color=i) for i in lm_colors],term_stats.index.values, 'upper right')

    fig.set_size_inches(12, 12)
    fig.savefig(os.path.join(os.path.dirname(__file__),'plots','BikeShareTrends.png'), transparent=True)
    plt.show()


    # term_stat

    # lm_st_counts = term_stats.groupby(['landmark'])['Start Terminal Count'].sum()
    # lm_dur_avg = term_stats.groupby(['landmark'])['Avg Duration'].sum()
    # lm_sub_perc = term_stats.groupby(['landmark'])['Subscriber Fraction']

def plotNodeSuccess(station, trip):
    # sort stations by distance for every station
    n_stations = 3;
    station['distance map'] = [ np.argsort([ (((station['lat'][j]-station['lat'][i])*degToKm)**2+((station['long'][j]-station['long'][i])*degToKm)**2)**0.5 for j in station.index.values ]) for i in station.index.values]

    # calculate average distance to three nearest neighbors
    station['avg dist to 3 nearest'] = [ sum([ (((station['lat'][j]-station['lat'][i])*degToKm)**2+((station['long'][j]-station['long'][i])*degToKm)**2)**0.5 for j in station['distance map'][i][1:n_stations+1] ]) / n_stations for i in station.index.values ]

    # calculate average elevation difference for three nearest neighbors
    station['avg elev dif to 3 nearest'] = [ sum([ station['elev'][j]-station['elev'][i] for j in station['distance map'][i][1:n_stations+1] ]) / n_stations for i in station.index.values ]

    # calculate std's away from mean elevation for each region
    lm_elev = station.groupby(['landmark'])['elev'].agg({'mean elev': np.mean, 'std elev': np.std})
    station['elev dif mean'] = station['elev'] - lm_elev['mean elev'][station['landmark']].values
    station['elev stds abv mean'] = station['elev dif mean'] / lm_elev['std elev'][station['landmark']].values

    # Get information about the number of arrivals and departures from each station
    trip_st = trip.groupby(['Start Terminal'])['Start Terminal'].count()
    trip_en = trip.groupby(['End Terminal'])['End Terminal'].count()
    trip_st = pd.DataFrame({'departures': trip_st}, trip_st.index.values)
    trip_en = pd.DataFrame({'arrivals': trip_en}, trip_en.index.values)
    st_dep_and_arr = pd.concat([trip_st, trip_en], axis=1)
    st_dep_and_arr['dep and arr'] = st_dep_and_arr['departures'] + st_dep_and_arr['arrivals']
    st_ind = [ station[station['station_id']==t].index[0] for t in st_dep_and_arr.index.values ]
    st_dep_and_arr.index = st_ind
    station = station.join(st_dep_and_arr)
    station['in operation td'] = pd.Timestamp(date_most_recent) - station['installation']

    # Remove stations with no traffic (only really applicable for very testing small sample size)
    station = station[~np.isnan(station['dep and arr'])]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(station['avg dist to 3 nearest'], station['avg elev dif to 3 nearest'], station['dep and arr'] / (station['in operation td'] / np.timedelta64(1, 'M')) * 0.001, s=80, c=station['elev stds abv mean'], cmap=cm.seismic)

    cb = plt.colorbar(sc)
    cb.set_label('Elevation Stds above Mean (per location) (m)')

    ax.set_xlabel('Avg. Distance to 3 Nearest (km)', fontsize=10)
    ax.set_ylabel('Avg. Elevation Change to 3 Nearest (m)', fontsize=10)
    ax.set_zlabel('Trips Per Month (1000s)', fontsize=10)
    ax.set_zlim(bottom=0)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # y_min, y_max = ax.get_ylim()
    # x_min, x_max = ax.get_xlim()
    # z_min, z_max = ax.get_zlim()

    # experimenting with adding a bit of a shadow projection... ended up not caring for it
    # ax.plot(station['avg dist to 3 nearest'], station['avg elev dif to 3 nearest'], linestyle='None', marker='o', mew=0, mfc=(0, 0, 0, 0.1), zdir='z', zs=0.01, zorder=-1)

    fig.set_size_inches(12, 12)
    fig.savefig(os.path.join(os.path.dirname(__file__),'plots','BikeShareTrafficWithDistElev.png'), transparent=True)
    plt.show()


def drawPlots(station_data, trip_data, lmstats):
    # Plot 1
    # plotStations(station_data, trip_data)

    # Plot 2
    #plotTransitPerCapita(station_data, trip_data, lmstats)

    # Plot 3
    plotNodeSuccess(station_data, trip_data)
    return

def getElevationFromAPI(station_data, key=KEY):
    elev = []
    res = []

    if allow_api_calls:
        for i in np.arange(len(station_data.index.values)):
            print(station_data['lat'][i], station_data['long'][i])
            request = urllib.request.urlopen('{}locations={},{}&key={}'.format(base_url,str(station_data['lat'][i]), str(station_data['long'][i]), key))
            jsonrequest = json.loads(request.read().decode())
            elev.append(jsonrequest['results'][0]['elevation'])
            res.append(jsonrequest['results'][0]['resolution'])

    return elev, res

# Pass Google Elevation API Key as -k argument
def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'k:', ['apikey='])
        for opt, arg in opts:
            if opt == '-h':
                print('babs-elev-api-calls.py -k <google_elevation_api_key>')
                sys.exit()
            elif opt in ('-k', '--apikey'):
                try:
                    KEY = str(arg)
                    print("Running with api key: ", KEY)
                finally:
                    break;
    except getopt.GetoptError:
        print('babs-elev-api-calls.py -k <google_elevation_api_key>')

    # some info I just looked up manually (various Wiki Pages) - stats from 2011-2013
    # https://en.wikipedia.org/wiki/List_of_California_locations_by_income
    lmstats = {'population': [939688, 76031, 73394, 63475, 797983],
               'per_cap_income': [33770, 39927, 51635, 72199, 46777],
               'household_median_income': [80764, 77111, 91446, 122532, 72947]}
    lmstats = pd.DataFrame(data=lmstats, index=['San Jose', 'Redwood City', 'Mountain View', 'Palo Alto', 'San Francisco'])

    station_data = LoadData(os.path.join(os.path.dirname(__file__),'generated_files'), date_format='%m/%d/%Y', date_headers=['installation'])

    weather_data = LoadData(os.path.join(os.path.dirname(__file__),'babs_archives','weather_data'), date_headers=['Date'], date_format='%m/%d/%Y', repstrs=((' ', ''), ('_', ''), ('PDT', 'Date'), ('zip', 'Zip'), ('In', '')))

    trip_data = LoadData(os.path.join(os.path.dirname(__file__),'babs_archives','trip_data'), date_headers=['Start Date', 'End Date'], date_format='%m/%d/%Y %H:%M', verbose=True, req_headers=['Duration','Start Date','Start Terminal','End Date','End Terminal','Subscription Type'])

    # This dataset is quite large without much meat in it, just leaving it mostly unloaded for now
    status_data = LoadData(os.path.join(os.path.dirname(__file__),'babs_archives','status_data'), nrows=30)

    # draw our plots
    drawPlots(station_data, trip_data, lmstats)

if __name__ == "__main__":
    main(sys.argv[1:])
