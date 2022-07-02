#!/bin/bash

if [ $# -eq 1 ] || [ $1 = kodak ]; then
    ssid="PIXPRO-SP360_82EF"
    password="12345678"
else
    ssid=$1
    password=$2
fi

nmcli radio wifi off
sleep 1
nmcli radio wifi on
sleep 3
nmcli device wifi connect ${ssid} password ${password}
