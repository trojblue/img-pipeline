import os
import os.path as opth
import csv

bin_dir = opth.join(opth.dirname(__file__), "..", "bin")


def load_artists():
    csv_path = opth.join(bin_dir, "artist_replacements_dbr.csv")
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        return data
