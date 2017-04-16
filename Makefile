
CXX=g++
CPPFLAGS=-std=c++11 -g
CILPATH=~/Code/obliv-c

CC=gcc
CFLAGS=-g -lstdc++ -lm
BASE_OBJS=party.o train.o privacy.o
YAO_OBJS=wrapper.o yao_util.o

all: gen train gradient_yao full_yao

gen: gen_main.o
	$(CXX) -o $@ $^

train:  train_main.o $(BASE_OBJS)
	$(CXX) -o $@ $^

gradient_yao: gradient_yao.oc gradient_yao.c $(BASE_OBJS) $(YAO_OBJS)
	$(CILPATH)/bin/oblivcc $(CFLAGS) -I . $^ -o gradient_yao

full_yao: full_yao.oc full_yao.c $(BASE_OBJS) $(YAO_OBJS)
	$(CILPATH)/bin/oblivcc $(CFLAGS) -I . $^ -o full_yao

yao_util.o: yao_util.c
	$(CILPATH)/bin/oblivcc $(CFLAGS) -I . $^ -c

clean:
	rm *.o gen train gradient_yao full_yao
