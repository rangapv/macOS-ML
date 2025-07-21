#!/usr/bin/env zsh
#author:rangapv@yahoo.com
#20-07-25

source <(curl -s https://raw.githubusercontent.com/rangapv/bash-source/main/s1.sh) >>/dev/null 2>&1
source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/refs/heads/main/pyverchk.sh) >>/dev/null 2>&1


#echo "$mac"
#echo "$una"

if [ ! -z "$mac" ]
then
	echo "It is a Mac"
	cm1="brew"
	count=1
else
	echo "This script is for MacOS  distribution only"
        exit
fi

pyvercheck python3

if [ "$pyuni" = "python3" ]
then
    echo "Then installed Python on this Mac is $pyuni with the version $piver1"
else
    echo "No python found"
fi
