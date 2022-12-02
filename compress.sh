#!/bin/bash
FOLD=$1
files=`find ${FOLD} -name "*best.pth" | sort`
for file in ${files}; 
do
	{
	zipfile=`basename $file`
	zip -jr ${zipfile}.zip ${file} &
	printf "\n"
	} #& 
done

wait
printf "Done!\n"

