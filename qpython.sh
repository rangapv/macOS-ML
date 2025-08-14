#!/usr/bin/env zsh
#author: rangapv@yahoo.com
#21-07-25

source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/refs/heads/main/pyverchk.sh) >/dev/null 2>&1
source <(curl -s https://raw.githubusercontent.com/rangapv/ansible-install/refs/heads/main/libraries.sh ) >/dev/null 2>&1

pyflg="0"
cm1="brew"

quick_python(){

arg1="$@"
if [ -z "$arg1" ]
then
qpv="3.13.5"
else
qpv="$arg1"
fi

pyvercheck python3
if [ -z "$pyuni" ]
then
   echo "No python present going to install version $qpv"
else
   echo "Current python is `${pyuni} --version`"
   pyflg="1"
   echo "If you requested a different version then press 'y'"
   read resp1
   if  [[ "$resp1" == "y" ]]
   then
     pyflg="0"
     sudo $cm1 -y install gcc make wget libffi-dev 
     zlibadd
     sslupdate $cm1 
     packages $cm1
   else
     exit
   fi
fi

echo "Installing Python version $qpv"
qp1=`sudo wget https://www.python.org/ftp/python/${qpv}/Python-${qpv}.tgz`
qp1s="$?"
if [ "$qp1s" = "0" ]
then
 qp2=`sudo tar -xvf ./Python-${qpv}.tgz`
 qp2s="$?"
 if [ "$qp2s" = "0" ]
 then
  qp3=`cd ./Python-${qpv}`
  qp3s="$?"
  if [ "$qp3s" = "0" ]
  then
   qp4=`sudo ./Python-${qpv}/configure`
   qp4s="$?"
   if [ "qp4s" = "0" ]
   then
    qp5=`sudo ./Python-${qpv}/make`
    qp5s="$?"
    if [ "$qp5s" = "0" ]
    then
     qp6=`sudo make ./Python-${qpv}/install`
     qp6s="$?"
     pyflg="1"
    fi
   fi
  fi
 fi
fi
}


pipinstall(){

pyvercheck python3
if [ -z "$pyuni" ]
then
   echo "No python present do install Python"
   exit
else
   echo "Current python is $pyuni"
   pyflg="1"
fi

list1=("mlx[cuda]","torch")

if [ "$pyflg" = "1" ]
then
   for i in ${list1[@]}:
   do
      pip1=`sudo pip3 install $i`
      pip1s="$?"
      if [ "$pip1s" != "0" ]
      then
        echo "trying broken system pacakge INSTALL for $i"
        pip2=`sudo pip3 install "$i" --break-system-packages`
      fi
   done
fi
}

quick_python 3.12.0

pipinstall

