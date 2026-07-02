###########################################################################################
#
#  File:    Makefile
#  Purpose: Local header-only build for the CCCL FP SDK unit tests (DRAFT).
#
#  This is a lightweight, self-contained build used for internal validation of the
#  FP SDK (cuda::experimental) unit tests. It compiles directly against the in-tree
#  CCCL headers -- <cuda/fpmp>, <cuda/fpemu>, <cuda/fptool> -- with no library
#  build or install step (the FP SDK is header-only). The eventual CCCL
#  CMake/CTest integration is a separate follow-up.
#
#                       USER GUIDE
#                       ==========
#
#  Parameters:
#    TARGET              = target backend: `device` (default, nvcc) | `host` (g++)
#    UNIT                = unit test(s) to build (default: all *.cpp), e.g. `UNIT=fpmp_api`
#    ARCH                = CUDA sm_xx (unset: detect via nvidia-smi, else 86)
#    OUT                 = output directory (default: `_out`)
#    VERBOSE             = `0` (silent, default) | `1` | `2` (full cmds) | `3` (very verbose)
#                          Aliases: `V`, `VERB`
#    PACKED_VIA_UNPACKED = `y`/`n` route packed fpemu builtins through unpacked cores (test mode)
#
#  Targets:
#    all / build  - build selected unit tests (produces *.exe in OUT)
#    run          - build and run selected unit tests (writes *.log)
#    rebuild      - clean then build
#    rerun        - clean then build and run
#    clean        - remove OUT
#    help         - print usage
#
#  Examples:
#    make                              # build all units (device)
#    make TARGET=host run              # build + run all on host (CPU)
#    make TARGET=host UNIT=fpmp_api run
#
###########################################################################################

# current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# CCCL libcudacxx headers (provides <cuda/fptool>, <cuda/fpmp>, <cuda/fpemu>, <cuda/std/...>)
CCCL_INCLUDE := $(THIS_DIR)/../../../../../include
INCLUDES     := -I$(CCCL_INCLUDE)

# defaults
TARGET  ?= device
OUT     ?= $(THIS_DIR)/_out

# Verbose mode: 0 (default, silent) | 1 (minimal) | 2 (full cmds) | 3 (very verbose).
# Aliases: V, VERB. At mode >= 2 the .SILENT directive is dropped so the actual
# g++/nvcc command lines are echoed.
VERBOSE_MODE = 0
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

# Packed-via-unpacked test mode: route the packed fpemu builtins through the
# unpacked cores (the unpacked API always co-exists). Opt-in; default off.
ifneq (,$(filter y Y yes Yes true True 1,$(PACKED_VIA_UNPACKED)))
    UNIFIED_FLAGS := -D__FPEMU_PACKED_VIA_UNPACKED__=1
else ifneq (,$(filter n N no No false False 0,$(PACKED_VIA_UNPACKED)))
    UNIFIED_FLAGS := -D__FPEMU_PACKED_VIA_UNPACKED__=0
else
    UNIFIED_FLAGS :=
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

OBJFLAGS  += $(UNIFIED_FLAGS) $(EXTRA_FLAGS)
LINKFLAGS += $(EXTRA_FLAGS)

# unit selection
ALL_UNITS := $(sort $(basename $(notdir $(wildcard $(THIS_DIR)/*.cpp))))
UNIT      ?= $(ALL_UNITS)
UNITS     := $(filter $(UNIT),$(ALL_UNITS))

EXE_FILES := $(addprefix $(OUT)/,$(addsuffix .exe,$(UNITS)))
OBJ_FILES := $(addprefix $(OUT)/,$(addsuffix .o,$(UNITS)))
LOG_FILES := $(addprefix $(OUT)/,$(addsuffix .log,$(UNITS)))

ifeq ($(VERBOSE_MODE),1)
    RULE_DONE = -@echo "  unit test built: $(notdir $@)"
else
    RULE_DONE :=
endif

# Very-verbose (mode 3): also ask the compiler/driver to be chatty.
ifeq ($(VERBOSE_MODE),3)
    OBJFLAGS  += -v
    LINKFLAGS += -v
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
	@echo "  Running unit test: $$(basename $<)"
	@echo "  =============================="
	@$< | tee $@

clean:
	rm -rf $(OUT)

help:
	@echo "CCCL FP SDK unit tests (local header-only build)"
	@echo ""
	@echo "Usage: make [TARGET=host|device] [UNIT=fpmp_api|fpemu_core|...] [ARCH=86] [OUT=...] [VERBOSE=0|1|2|3] (aliases: V, VERB) [PACKED_VIA_UNPACKED=y|n]"
	@echo ""
	@echo "Targets: all/build, run, rebuild, rerun, clean, help"
	@echo ""
	@echo "Available units:"
	@echo "  $(ALL_UNITS)"
	@echo ""
	@echo "Notes:"
	@echo "  - Builds directly against CCCL headers: -I$(CCCL_INCLUDE)"
	@echo "  - Header-only: no library build/install step."
	@echo "  - libquadmath auto-detect (x86 host): HAS_QUADMATH=$(HAS_QUADMATH). Override with HAS_QUADMATH=1/0."

.PHONY: all build run rebuild rerun clean help

# Silence recipe echoing for the quiet modes; at mode >= 2 leave it on so the
# compiler command lines are visible.
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif