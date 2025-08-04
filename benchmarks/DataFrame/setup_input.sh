#!/bin/bash

# The total input file size is about 16 GB.
# The max working set size in memory is about 31 GiB.
head=yellow_tripdata_2016-01.csv

sudo umount /dev/sda4
sudo mkfs.ext4 -F /dev/sda4
sudo mount /dev/sda4 /mnt
sudo chmod a+rw /mnt
cp -r /proj/farmemory-PG0/dataset_atlas/NYC-yellow-tripdata/. /mnt
cd /mnt

cat $head > all.csv

for file in `ls *.csv`
do
    if [ "$file" = "all.csv" ]; then
	continue
    fi
    if [ "$file" = "$head" ]; then
	continue
    fi
    awk '{if (NR > 1) print $0}' $file >> all.csv
done
