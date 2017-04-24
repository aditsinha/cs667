
CXX=g++
CXXFLAGS=-std=c++11 -g

CC=gcc
CFLAGS=-g -lstdc++ -lm

CILPATH=~/Code/obliv-c
OCC=$(CILPATH)/bin/oblivcc

OC_SRCS=full_yao.c yao_util.c gradient_yao.c obliv_math_def.c 
CC_SRCS=full_yao_simulator.cc party.cc train.cc privacy.cc train_main.cc

BASE_OBJS=party.o train.o privacy.o
YAO_OBJS=wrapper.o yao_util.o
OBLIV_MATH_OBJS=obliv_math_def.o obliv_math_func.h

all: depend gen train gradient_yao full_yao full_yao_simulator format_mnist

depend: .depend

.depend: $(OC_SRCS) $(CC_SRCS)
	rm -f ./.depend
	$(CXX) $(CPPFLAGS) -MM $(CC_SRCS) >> .depend
	$(OCC) $(CFLAGS) -MM $(OC_SRCS) >> .depend

include .depend

gen: gen_main.o
	$(CXX) -o $@ $^

train:  train_main.o $(BASE_OBJS)
	$(CXX) -o $@ $^

gradient_yao: gradient_yao.oc gradient_yao.c $(BASE_OBJS) $(YAO_OBJS)
	$(OCC) $(CFLAGS) -I . $^ -o gradient_yao

full_yao: full_yao.oc full_yao.c $(BASE_OBJS) $(YAO_OBJS) $(OBLIV_MATH_OBJS)
	$(OCC) $(CFLAGS) -I . $^ -o full_yao

full_yao_simulator: full_yao_simulator.o $(BASE_OBJS) $(OBLIV_MATH_OBJS)
	$(CXX) -o $@ $^

yao_util.o: yao_util.c
	$(OCC) $(CFLAGS) -I . $^ -c

clean:
	rm *.o gen train gradient_yao full_yao full_yao_simulator ./.depend
