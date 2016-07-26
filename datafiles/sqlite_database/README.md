# Contains Aggregated Data Per Station

## Database Construction
* Sources: BABS station_data and trip_data
* Aggregated counts of trips originating and ending at each station by "Station ID"
* Joined with elevation using Google's Elevation API

## Columns
1. "Station ID" - Numeric ID for station
2. "Latitude" - Latitude of station
3. "Longitude" - Longitude of station
4. "Dock Count" - Number of docks available at station
5. "Landmark" - Written name for station
6. "Installation" - Date of installation
7. "Start Count" - Number of rides originating from station
8. "End Count" - Number of rides terminating at this station
9. "Elevation" - Elevation of station
10. "Elevation Resolution" - Resolution of elevation data
