from ultralytics import YOLO
import requests
import json
import shutil
import os
import math

####################################################################
# Define constants
#

# The Panoramax API endpoint to use
PANORAMAX_API="https://api.panoramax.xyz/api"
# The search area (min X, min Y, max X, max Y)
SEARCH_BBOX=[2.25256,48.96895,2.26447,48.97247]
# Path to your trained model ".pt" file
MODEL_PATH="hydrants_model_v1/train/weights/best.pt"
# Output file name for GeoJSON of detected features
OUTPUT_GEOJSON="./detected_features.geojson"
# Output folder for pictures showing detected features
OUTPUT_PICTURES="./detected_features_pictures"
# How many pictures should be tested at once
PICS_CHUNK_SIZE=10
# Amount of sequences per API call
NB_SEQUENCES_API=1000
# Class ID to target in detections
CLASS_ID=0
# Object name (only for display)
OBJECT_NAME="Hydrant"


############################################################################
# Function to process a chunk of images
#  This allows faster processing
#

print("Loading detection model...")
model = YOLO(MODEL_PATH)

def processPicturesChunk(items, start, end):
	print(f"      - Processing pictures {start+1} to {min(len(items), end)} / {len(items)}")
	chunk = [ i["assets"]["sd"]["href"] for i in items[start:end] ]
	results = model.predict(source=chunk, imgsz=2048, stream=True, max_det=1, classes=[CLASS_ID], verbose=False)
	i = 0
	picResults = []
	for res in results:
		# If a picture has detections, save it
		if len(res.boxes) > 0:
			item = items[start+i]
			print("        -", OBJECT_NAME, "found in picture", item["id"])
			picResults.append(item)
			res.save(filename=f"{OUTPUT_PICTURES}/{item['id']}.jpg")

			# Also save original in case it's a false positive
			picResponse = requests.get(item["assets"]["sd"]["href"], stream=True)
			if picResponse.status_code == 200:
				with open(f"{OUTPUT_PICTURES}/{item['id']}.orig.jpg", 'wb') as f:
					picResponse.raw.decode_content = True
					shutil.copyfileobj(picResponse.raw, f)

		i+=1
	return picResults


############################################################################
# Function to handle a single page of sequences from API
#

def processAPISequencePage(url, pageNb = 1):
	global picsWithFeatures
	pnmxCollectionsResponse = requests.get(url)
	pnmxCollections = pnmxCollectionsResponse.json()

	# Reading downloaded metadata
	nbSequences = len(pnmxCollections["collections"])
	nextLink = next((l["href"] for l in pnmxCollections["links"] if l["rel"] == "next"), None)
	hasNextPage = nextLink is not None and nbSequences == NB_SEQUENCES_API
	if hasNextPage:
		if pageNb == 1:
			print("More than", nbSequences, "found, processing page", pageNb)
		else:
			print("Processing page", pageNb, "with", nbSequences, "sequences")
	else:
		print(nbSequences, "sequences found")
	
	for i, collection in enumerate(pnmxCollections["collections"], start=1):
		# List pictures in this collection
		try:
			print("  - Find pictures in sequence", collection["id"], f"({i}/{nbSequences} - {math.floor(i/nbSequences*100)}%)")
			pnmxCollectionItemsResponse = requests.get(f"{PANORAMAX_API}/collections/{collection['id']}/items")
			pnmxCollectionItems = pnmxCollectionItemsResponse.json()

			# Only keep pictures really in search bounding box
			picturesInBbox = []
			for i in pnmxCollectionItems["features"]:
				lon, lat = i["geometry"]["coordinates"]
				if (
					lon >= SEARCH_BBOX[0]
					and lon <= SEARCH_BBOX[2]
					and lat >= SEARCH_BBOX[1]
					and lat <= SEARCH_BBOX[3]
				):
					picturesInBbox.append(i)
			
			# Run prediction over pictures, chunk by chunk
			if len(picturesInBbox) > 0:
				print(f"    - Detecting objects in {len(picturesInBbox)} pictures...")
				for c in range(0, len(picturesInBbox), PICS_CHUNK_SIZE):
					picsWithFeatures += processPicturesChunk(picturesInBbox, c, c+PICS_CHUNK_SIZE)
			else:
				print("    - Skipping sequence, no picture in search area")
		except Exception as e:
			print("    - Error during processing\n      ", e)

	if hasNextPage:
		processAPISequencePage(nextLink, pageNb + 1)


############################################################################
# Checking collections one by one
#

# Create output directory
shutil.rmtree(OUTPUT_PICTURES, ignore_errors=True)
os.mkdir(OUTPUT_PICTURES)

# Download collections from API
picsWithFeatures = []
print("List sequences in Panoramax...")
processAPISequencePage(f"{PANORAMAX_API}/collections?limit={NB_SEQUENCES_API}&bbox={','.join([ str(f) for f in SEARCH_BBOX])}")


############################################################################
# Save results
#

print(f"Exporting {len(picsWithFeatures)} features as GeoJSON...")
outputGeojson = {
	"type": "FeatureCollection",
	"features": picsWithFeatures
}
with open(OUTPUT_GEOJSON, "w") as outputFile:
	outputFile.write(json.dumps(outputGeojson))
	outputFile.close()

print("Done !")
