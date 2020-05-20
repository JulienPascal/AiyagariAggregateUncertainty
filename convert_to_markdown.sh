#!/bin/bash
echo "CONVERTING TO MARKDOW"
jupyter nbconvert --to markdown $PWD/AiyagariBKM.ipynb
echo "DONE"

