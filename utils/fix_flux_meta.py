#!/usr/bin/env python3
"""Fix Flux meta file format."""
import csv

# Create new meta file with proper format
with open('features/flux/meta_00000.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['img_id', 'label', 'dataset', 'transform'])
    for i in range(3000):
        writer.writerow([f'flux_{i}', 1, 'flux', 'clean'])

print("Created meta file with 3000 rows")
