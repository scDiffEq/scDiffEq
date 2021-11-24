#!/bin/bash

ACCESSIONS=$(cat SRA.ACC.LIST.PRJNA472237.txt)

for SRR in ${ACCESSIONS}
do
  fasterq-dump ${SRR} --split-files
done
