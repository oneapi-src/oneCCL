
ENABLE_DEBUG ?= 1
SHELL = bash

MLSL_INCLUDES = -I$(MLSL_ROOT)/intel64/include
MLSL_LDFLAGS  = -L$(MLSL_ROOT)/intel64/lib -lmlsl

CCL_INCLUDES  = -I$(CCL_ROOT)/include
CCL_LDFLAGS   = -L$(CCL_ROOT)/lib -lccl

CXX     ?= g++

ifeq ($(ENABLE_DEBUG),1)
  CFLAGS += -O0 -g
else
  CFLAGS += -O2
endif

MLSL_CXX_COMPILE = $(CXX) $(CFLAGS) $(MLSL_INCLUDES)
CCL_CXX_COMPILE = $(CXX) $(CFLAGS) $(CCL_INCLUDES)
# tests: mlsl_sample ccl_sample
tests       =  $(patsubst %.cpp, %.out, $(wildcard *.cpp))
all: $(tests)

mlsl_sample.out:
	$(MLSL_CXX_COMPILE) -o mlsl_sample.out mlsl_sample.cpp $(MLSL_LDFLAGS)
ccl_sample.out:
	$(CCL_CXX_COMPILE) -o ccl_sample.out ccl_sample.cpp $(CCL_LDFLAGS)


run: all
	for test in $(tests); do \
		echo "running $$test ..." ; \
		mpiexec.hydra -n 4 -ppn 1 -l ./$$test; \
		if [ "$${PIPESTATUS[0]}" -ne 0 ]; then \
			echo "Run FAILED." ; \
		fi ; \
	done;

clean:
	-rm -f $(tests)
