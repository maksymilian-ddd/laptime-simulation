import csv
from pyproj import Proj, transform

def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    proj_wgs84 = Proj(proj='latlong', datum='WGS84')
    proj_aeqd = Proj(proj='aeqd', lat_0=origin_lat, lon_0=origin_lon, datum='WGS84')
    x, y = transform(proj_wgs84, proj_aeqd, lon, lat)
    return x, y

def main(input_csv, output_csv):
    rows = []
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Skip first 9 header lines
        for _ in range(9):
            next(reader)
        # Read header row
        header = next(reader)
        # Read the rest of the rows
        for row in reader:
            rows.append(row)

    # Remove first two rows as in the pandas version
    rows = rows[2:]

    # Find column indices
    lap_idx = header.index('lap_number')
    lat_idx = header.index('latitude')
    lon_idx = header.index('longitude')

    # Filter for lap 116 only
    filtered_rows = [row for row in rows if row[lap_idx] == '116']

    # Extract lat/lon
    latitudes = [float(row[lat_idx]) for row in filtered_rows]
    longitudes = [float(row[lon_idx]) for row in filtered_rows]

    if not latitudes or not longitudes:
        print("No data for lap 116.")
        return

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
            if i % 9 == 0:
                writer.writerow([f"{x:.6f}", f"{y:.6f}"])

input_csv = r"C:\Users\piese\Desktop\moje\ARC\Slovakia Ring 2025\session_20250627_085949_slovakiaring_4_1_v3.csv"
output_csv = "raceline_SlovakiaRing.csv"

main(input_csv, output_csv)
