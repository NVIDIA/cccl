###########################################################################################
#
#  File:    Makefile
#  Purpose: Local header-only build for the CCCL FP SDK examples (DRAFT).
#
#  This is a lightweight, self-contained build used for internal validation of the
#  FP SDK (cuda::experimental) examples. It compiles directly against the in-tree
#  CCCL headers -- <cuda/fpmp>, <cuda/fpemu>, <cuda/fptool> -- with no library
#  build or install step (the FP SDK is header-only). The eventual CCCL
#  CMake/CTest integration is a separate follow-up.
#
#                       USER GUIDE
#                       ==========
#
#  Parameters:
#    TARGET   = target backend: `device` (default, nvcc) | `host` (g++)
#    EXAMPLE  = example(s) to build (default: all *.cpp), e.g. `EXAMPLE=fp32mp2`
#    ARCH     = CUDA sm_xx (unset: detect via nvidia-smi, else 86)
#    OUT      = output directory (default: `_out`)
#    VERBOSE  = `0` (silent) | `1` (default) | `2` (full cmds) | `3` (very verbose)
#               Aliases: `V`, `VERB`
#
#  Targets:
#    all / build  - build selected examples (produces *.exe in OUT)
#    run          - build and run selected examples (writes *.log)
#    rebuild      - clean then build
#    rerun        - clean then build and run
#    clean        - remove OUT
#    help         - print usage
#
#  Examples:
#    make                                   # build all examples (device)
#    make TARGET=host run                   # build + run all on host (CPU)
#    make TARGET=host EXAMPLE=fp32mp2 run   # build + run a single example on host
#
###########################################################################################

# current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# CCCL libcudacxx headers (provides <cuda/fptool>, <cuda/fpmp>, <cuda/fpemu>, <cuda/std/...>)
CCCL_INCLUDE := $(THIS_DIR)/../../libcudacxx/include
INCLUDES     := -I$(CCCL_INCLUDE)

# defaults
TARGET  ?= device
OUT     ?= $(THIS_DIR)/_out

# Verbose mode: 0 (silent) | 1 (default, minimal) | 2 (full cmds) | 3 (very verbose).
# Aliases: V, VERB. At mode >= 2 the .SILENT directive is dropped so the actual
# g++/nvcc command lines are echoed.
VERBOSE_MODE = 1
ifneq (,$(filter 0 n N no No,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 0
else ifneq (,$(filter 1 y Y yes Yes,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 1
else ifneq (,$(filter 2,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 2
else ifneq (,$(filter 3 full Full f F,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 3
endif

# CUDA arch: use ARCH if set; else first GPU (nvidia-smi); else 86.
ifndef ARCH
    _fp_arch := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
    ARCH     := $(if $(strip $(_fp_arch)),$(_fp_arch),86)
endif

# libquadmath is only shipped by x86 GCC distributions; needed when the FP128 host
# math fallback is enabled. Override with HAS_QUADMATH=1/0 to bypass auto-detection.
HOST_ARCH := $(shell uname -m 2>/dev/null)
ifndef HAS_QUADMATH
    ifneq (,$(filter $(HOST_ARCH),x86_64 i386 i686 amd64))
        HAS_QUADMATH := 1
    else
        HAS_QUADMATH := 0
    endif
endif
ifeq ($(HAS_QUADMATH),1)
    QUADMATH_LIB := -lquadmath
else
    QUADMATH_LIB :=
endif

# toolchain / flags
ifeq ($(TARGET),host)
    CXX       := g++
    OBJFLAGS  := -std=c++17 -O2 -c
    LINKFLAGS :=
    LINKLIBS  := $(QUADMATH_LIB)
else
    CXX       := nvcc
    OBJFLAGS  := -std=c++17 -arch=sm_$(ARCH) -O2 -x cu -dc -Xcompiler -Wall,-Wextra -Xcompiler -Wno-strict-aliasing
    LINKFLAGS := -rdc=true -arch=sm_$(ARCH)
    LINKLIBS  := $(QUADMATH_LIB)
endif

OBJFLAGS  += $(EXTRA_FLAGS)
LINKFLAGS += $(EXTRA_FLAGS)

# example selection
ALL_EXAMPLES := $(sort $(basename $(notdir $(wildcard $(THIS_DIR)/*.cpp))))
EXAMPLE      ?= $(ALL_EXAMPLES)
EXAMPLES     := $(filter $(EXAMPLE),$(ALL_EXAMPLES))

EXE_FILES := $(addprefix $(OUT)/,$(addsuffix .exe,$(EXAMPLES)))
OBJ_FILES := $(addprefix $(OUT)/,$(addsuffix .o,$(EXAMPLES)))
LOG_FILES := $(addprefix $(OUT)/,$(addsuffix .log,$(EXAMPLES)))

ifeq ($(VERBOSE_MODE),1)
    RULE_DONE = -@echo "  example built: $(notdir $@)"
else
    RULE_DONE :=
endif

# Very-verbose (mode 3): also ask the compiler/driver to be chatty.
ifeq ($(VERBOSE_MODE),3)
    OBJFLAGS  += -v
    LINKFLAGS += -v
endif

.PHONY: all build run rebuild rerun clean help

# Silence recipe echoing for the quiet modes; at mode >= 2 leave it on so the
# compiler command lines are visible.
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif

all: build

build: $(EXE_FILES)

run:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build
	rm -f $(LOG_FILES)
	MAKEFLAGS= $(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile $(LOG_FILES) -j1

rebuild:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build

rerun:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile run

$(OUT):
	mkdir -p $(OUT)

$(OBJ_FILES): $(OUT)/%.o: $(THIS_DIR)/%.cpp | $(OUT)
	$(CXX) $(OBJFLAGS) $(INCLUDES) $< -o $@
	$(RULE_DONE)

$(EXE_FILES): $(OUT)/%.exe: $(OUT)/%.o | $(OUT)
	$(CXX) $(LINKFLAGS) $< -o $@ $(LINKLIBS)
	$(RULE_DONE)

$(LOG_FILES): $(OUT)/%.log: $(OUT)/%.exe | $(OUT)
	@echo ""
	@echo "  Running example: $$(basename $<)"
	@echo "  =============================="
	@$< | tee $@

clean:
	rm -rf $(OUT)

help:
	@echo "CCCL FP SDK examples (local header-only build)"
	@echo ""
	@echo "Usage: make [TARGET=host|device] [EXAMPLE=fp32mp2|fpemu|...] [ARCH=86] [OUT=...] [VERBOSE=0|1|2|3] (aliases: V, VERB)"
	@echo ""
	@echo "Targets: all/build, run, rebuild, rerun, clean, help"
	@echo ""
	@echo "Available examples:"
	@echo "  $(ALL_EXAMPLES)"
	@echo ""
	@echo "Notes:"
	@echo "  - Builds directly against CCCL headers: -I$(CCCL_INCLUDE)"
	@echo "  - Header-only: no library build/install step."
	@echo "  - libquadmath auto-detect (x86 host): HAS_QUADMATH=$(HAS_QUADMATH). Override with HAS_QUADMATH=1/0."
