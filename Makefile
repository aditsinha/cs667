
CXX=g++
CPPFLAGS=-std=c++11 -g

all: gen train

gen: gen_main.o
	$(CXX) -o $@ $^

train:  train_main.o party.o train.o privacy.o
	$(CXX) -o $@ $^
