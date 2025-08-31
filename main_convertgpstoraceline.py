import csv
from pyproj import Proj, transform

def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    proj_wgs84 = Proj(proj='latlong', datum='WGS84')
    proj_aeqd = Proj(proj='aeqd', lat_0=origin_lat, lon_0=origin_lon, datum='WGS84')
    x, y = transform(proj_wgs84, proj_aeqd, lon, lat)
    return x, y

def main(input_csv, output_csv, racechrono_or_motec, downsampling_rate):
    if racechrono_or_motec == 'motec':
        header_lines_to_skip = 14
        rows_to_skip = 3
    elif racechrono_or_motec == 'racechrono':
        header_lines_to_skip = 8
        rows_to_skip = 2

    rows = []
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Skip first n header lines
        for _ in range(header_lines_to_skip):
            next(reader)
        # Read header row
        header = next(reader)
        # Read the rest of the rows
        for row in reader:
            rows.append(row)

    # Remove first two rows as in the pandas version
    rows = rows[rows_to_skip:]
    print(rows)

    # Find column indices
    # lap_idx = header.index('lap_number')
    print(header)
    lat_idx = header.index('GPS Latitude')
    lon_idx = header.index('GPS Longitude')

    # Filter for lap 116 only
    # filtered_rows = [row for row in rows if row[lap_idx] == '116']
    filtered_rows = [row for row in rows]
    print(filtered_rows[0])
    print(filtered_rows[len(filtered_rows) - 1])


    # Extract lat/lon
    latitudes = []
    longitudes = []
    for row in filtered_rows:
        if row[lat_idx] != '' and row[lon_idx] != '':
            latitudes.append(row[lat_idx])
            longitudes.append(row[lon_idx])

    origin_lat = latitudes[0]
    origin_lon = longitudes[0]

    xy_points = []
    for lat, lon in zip(latitudes, longitudes):
        x, y = latlon_to_xy(lat, lon, origin_lat, origin_lon)
        xy_points.append((x, y))

    with open(output_csv, 'w', newline='') as csvfile:
        csvfile.write("# x_m,y_m\n")
        writer = csv.writer(csvfile)
        for i, (x, y) in enumerate(xy_points):
            if i % downsampling_rate == 0:
                writer.writerow([f"{x:.6f}", f"{y:.6f}"])

input_csv = r"C:\Users\piese\Desktop\moje\ARC\#3 - Le Most 6H\31082025-195542-Maksymilian Dąbrowski-bmw_e36_compact-rt_autodrom_most.csv"
output_csv = "raceline_AutodromMostAC.csv"
racechrono_or_motec = 'motec'
downsampling_rate = 1

main(input_csv, output_csv, racechrono_or_motec, downsampling_rate)
