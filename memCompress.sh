#!/bin/bash
FOLD=$1
MEMFOLD="/mnt/ramdisk/memzip"
mkdir -p ${MEMFOLD}

## ColorPrint parameters: 	$1 -- print content
##				$2 -- print color 0(red) 1(green) 2(yellow) 4(blue)
ColorPrint()
{
	line=$1
	
	case ${2} in
		0)
			front_color=31
			;;
		1)
			front_color=32
			;;
		2)
			front_color=33
			;;
		4)	
			front_color=34
			;;
		*)
			front_color=36 #35
			;;
	esac

	echo -n -e "\e[${front_color};1m${line}\e[0m"
	return 0
}

echo ""
ColorPrint "Cleaning history files... ..., all files in ${MEMFOLD} will be removed!\n" 0
#rm -rf ${MEMFOLD}/*
ColorPrint "Done!\n" 1

echo ""
ColorPrint "Copy file to memory disk... ...\n" 2
#cp `basename ${FOLD}`/*best.pth ${MEMFOLD} -vaf
rsync -avPh `basename ${FOLD}`/*best.pth ${MEMFOLD}
ColorPrint "Done!\n" 1

echo ""
files=`find ${MEMFOLD} -name "*best.pth" | sort`
ColorPrint "Compressing... ... ...\n" 4
for file in ${files}; 
do
	{
	zipfile=`basename $file`.zip
	if [ ! -f `basename ${FOLD}`/${zipfile} ]; then
		zip -jr ${MEMFOLD}/${zipfile} ${file} &
		printf "\n"
	else
		echo "skip zipping ${zipfile}"
	fi
	} #& 
done

wait
ColorPrint "Done!\n" 1

echo ""
mv ${MEMFOLD}/*.zip ${FOLD} -n -v
ColorPrint "All Finished!! ^_^ \n" 10


