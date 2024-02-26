import json
import requests
import shutil
import os

####################################################################
# Define constants
#

# The Panoramax API endpoint to use
PANORAMAX_API="https://panoramax.ign.fr/api"
# The GeoJSON input file
OSM_FEATURES="./osm_hydrants_lyon.geojson"
# How many pictures you want
WANTED_PICTURES=100
# Where to save pictures
PICTURES_OUTPUT_FOLDER="./training_pictures"


####################################################################
# Read features coordinates from OSM files
print("Loading OSM features...")
osmFeaturesCoordinates = []
with open(OSM_FEATURES) as osmFile:
	osmData = json.load(osmFile)
	for osmFeature in osmData["features"]:
		osmFeaturesCoordinates.append(osmFeature["geometry"]["coordinates"])
	osmFile.close()

print("  - Found", len(osmFeaturesCoordinates), "features")

# Call Panoramax "Show me" API
print("Fetching metadata from Panoramax...")
pnmxPicturesUrls = []
for coords in osmFeaturesCoordinates:
	pnmxResponse = requests.get(f"{PANORAMAX_API}/search?place_distance=2-10&place_position={coords[0]},{coords[1]}")
	pnmxResponseJson = pnmxResponse.json()

	# Only keep first one, should be the best match
	if len(pnmxResponseJson["features"]) > 0:
		print("  - It's a match @", coords)
		pnmxPicturesUrls.append((
			pnmxResponseJson["features"][0]["id"],
			pnmxResponseJson["features"][0]["assets"]["sd"]["href"]
		))

		if len(pnmxPicturesUrls) >= WANTED_PICTURES:
			break

# Download pictures
print("Downloading pictures from Panoramax...")
shutil.rmtree(PICTURES_OUTPUT_FOLDER, ignore_errors=True)
os.mkdir(PICTURES_OUTPUT_FOLDER)
for picId, picUrl in pnmxPicturesUrls:
	print("  - Loading", picUrl)
	picResponse = requests.get(picUrl, stream=True)
	if picResponse.status_code == 200:
		with open(f"{PICTURES_OUTPUT_FOLDER}/{picId}.jpg", 'wb') as f:
			picResponse.raw.decode_content = True
			shutil.copyfileobj(picResponse.raw, f)

print("Done !")
