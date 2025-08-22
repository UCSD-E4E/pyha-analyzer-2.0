
"""
Script to retrieve relevant templates for species from Gorongosa National Park, Mozambique.
The iNaturalist API is used to gather species from Gorongosa National Park.
Then, these species are used to query Xeno-Canto for up to 10 audio recordings per species and download them into folders.
"""

import os
import requests
import time

INAT_URL = "https://api.inaturalist.org/v1/taxa"
GORONGOSA_PLACE_ID = 15147
XC_URL = "https://www.xeno-canto.org/api/2/recordings"
DOWNLOAD_DIR = "/home/super/data/music/gorongosa_xc_templates"
MAX_CALLS = 10

def get_gorongosa_species():
	params = {"place_id": GORONGOSA_PLACE_ID, "verifiable": "true", "rank": "species", 
		"per_page": 200, "page": 1, "taxon_id": 1}
	species = set()
	while True:
		for attempt in range(5):
			try:
				r = requests.get(INAT_URL, params=params)
				if r.status_code in (429, 403):
					print(f"Rate limited (status {r.status_code}), sleeping 10s...")
					time.sleep(10)
					continue
				r.raise_for_status()
				break
			except requests.exceptions.RequestException as e:
				print(f"Request failed: {e}, retrying in 10s...")
				time.sleep(10)
		else:
			print("Failed too many times, aborting.")
			break
		results = r.json()
		for s in results["results"]:
			if s.get("name"):
				species.add(s["name"])
		if results["page"] * results["per_page"] >= results["total_results"]:
			break
		params["page"] += 1
		time.sleep(1)
	return list(species)

def query_xc_for_species(scientific_name, max_calls=10):
	parts = scientific_name.split()
	if len(parts) < 2:
		return []
	genus, species = parts[0], parts[1]
	query = f'gen:{genus} sp:{species}'
	params = {"query": query}
	r = requests.get(XC_URL, params=params)
	r.raise_for_status()
	data = r.json()
	return data.get("recordings", [])[:max_calls]

def download_xc_recording(rec, species_dir):
	url = f'https:{rec["file"]}' if rec["file"].startswith("//") else rec["file"]
	fname = os.path.join(species_dir, f'{rec["id"]}_{rec["gen"]}_{rec["sp"]}.mp3')
	if not os.path.exists(fname):
		try:
			resp = requests.get(url, timeout=30)
			resp.raise_for_status()
			with open(fname, 'wb') as f:
				f.write(resp.content)
			print(f"Downloaded: {fname}")
		except Exception as e:
			print(f"Failed to download {url}: {e}")
	else:
		print(f"Already exists: {fname}")

def main():
	os.makedirs(DOWNLOAD_DIR, exist_ok=True)
	print("Gathering list of species from iNaturalist.")
	species_list = get_gorongosa_species()
	print(f"Found {len(species_list)} species from iNaturalist.")

	for i, species in enumerate(species_list):
		print(f"[{i+1}/{len(species_list)}] Querying Xeno-Canto for species {species}")
		recs = query_xc_for_species(species, MAX_CALLS)
		if not recs:
			print(f"  No recordings found.")
			continue
		species_dir = os.path.join(DOWNLOAD_DIR, species.replace(' ', '_'))
		os.makedirs(species_dir, exist_ok=True)
		for rec in recs:
			download_xc_recording(rec, species_dir)
			time.sleep(1)

if __name__ == "__main__":
	main()