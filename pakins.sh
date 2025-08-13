#!/usr/bin/env bash
#author:rangapv@yahoo.com
#13-08-25

pakins() {

pakins1=`pip3 install -U $1  2>&1`
pakins1s="$?"
if [[ "$pakins1s" != "0" ]]
then
   pakins11=`echo "$pakins1" | grep "error: externally-managed-environment"`
   pakins12="$?"
   if [[ ! -z "$pakins11" ]]
   then
       pakins2=`pip3 install -U $1 --break-system-packages`
       pakins2s="$?"
       if  [[ "$pakins2s" == "0" ]]
       then
          echo "Python package '$line' from the requirements file installed SUCCESSFULLY"
       fi
   else
       echo "Tried both normal Install and Extenally-managed-envi Install unSUCCESSFULLY"
   fi
else
   echo "Python package from the requirements file $line installed SUCCESSFULLY"
fi
}

echo "This program will install the Python packages from the Requirements file \'requirements.txt\'"
echo "To input a different requirement FILE name press 'y'"
read req1

if  [[ "$req1" == "y" ]]
then
   echo "pls enter the file name containing the Python dependecies to be installed..."
   read reqfile1
else
   reqfile1="requirements.txt"
fi

while IFS= read -r line;do
  pakins $line
done < "$reqfile1" 
