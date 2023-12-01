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

# 或者 $(MAKE) -C cc
cc:
	cd cc && $(MAKE)

all: cc

clean: cc
	cd cc && $(MAKE) clean

.PHONY: all clean cc
