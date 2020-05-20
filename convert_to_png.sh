#!/bin/bash
for file in $PWD/AiyagariBKM_files/*.svg
do
     /usr/bin/inkscape -z -f "${file}" -w 640 -e "${file}.png"
done
