#!/bin/bash

if [ -z $1 ]; then
	echo "Please tell me the file name in command parameter."
	exit 1
fi

if [ -f $1 ]; then
  sed -i"" -e 's/[\(\)\,]/ /g' $1
else
  echo "The specified file doesn't exist or is not a regular file."
fi
