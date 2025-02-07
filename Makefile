PROJECT=Cecelila
VERSION=0.0.1

CC=gcc
CXX=g++
STDC=c11
STDCXX=c++11

CFLAGS=-std=$(STDC) -Wall
CXXFLAGS=-std=$(STDCXX) -Wall

ROOT=$(shell pwd)

export CC CXX CFLAGS CXXFLAGS ROOT

all: cc linux

# 或者 $(MAKE) -C cc
cc:
	cd cc && $(MAKE)

linux:
	cd linux && $(MAKE)

clean: cc linux
	cd cc && $(MAKE) clean
	cd linux && $(MAKE) clean

.PHONY: all clean cc linux
