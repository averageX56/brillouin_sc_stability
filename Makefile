# Brillouin cascade SDE solver.
#
#   make               optimised build with OpenMP   -> build/sde_solver[.exe]
#   make no-omp        single-threaded (no OpenMP in the toolchain)
#   make debug         -O1 + address/UB sanitisers   -> build/sde_solver_debug[.exe]
#   make drift_probe   test helper (drift on stdin)  -> build/drift_probe[.exe]
#   make NATIVE=1      add -march=native (faster, not portable)
#   make clean         remove build/
#   make distclean     also remove data/ (generated JSON sweeps and caches)
#
# Windows notes
# -------------
#  * The binary gets an .exe suffix automatically (detected via the OS variable),
#    so make's dependency tracking works instead of relinking on every call.
#    MinGW g++ appends .exe by itself, which is why a Unix-style target name
#    made every `make` call rebuild.
#  * Directory creation and deletion go through Python rather than shell
#    builtins: `mkdir -p` / `rm -rf` do not exist in cmd.exe, while
#    `if not exist` / `rmdir /s` do not exist in MSYS or Git Bash. Python works
#    in all of them. If your interpreter is not called `python`: make PY=py
#  * With MinGW the make program is often called `mingw32-make`.
#    No make at all? Run build.bat instead (see README).

CXX      ?= g++
PY       ?= python
NATIVE   ?= 0
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fno-math-errno
ifeq ($(NATIVE),1)
CXXFLAGS += -march=native
endif
OMPFLAGS ?= -fopenmp

ifeq ($(OS),Windows_NT)
EXE := .exe
else
EXE :=
endif

SRCDIR   := src
BUILDDIR := build
DATADIR  := data

TARGET   := $(BUILDDIR)/sde_solver$(EXE)
DEBUGBIN := $(BUILDDIR)/sde_solver_debug$(EXE)
PROBE    := $(BUILDDIR)/drift_probe$(EXE)

MAIN     := $(SRCDIR)/main.cpp
PROBESRC := $(SRCDIR)/drift_probe.cpp
HDR      := $(SRCDIR)/model.hpp $(SRCDIR)/rng.hpp $(SRCDIR)/solver.hpp \
            $(SRCDIR)/estimators.hpp $(SRCDIR)/linear_theory.hpp

# Portable "mkdir -p" / "rm -rf" (see the Windows notes above).
MKDIRS = $(PY) -c "import pathlib,sys;[pathlib.Path(p).mkdir(parents=True,exist_ok=True) for p in sys.argv[1:]]"
RMTREE = $(PY) -c "import shutil,sys;[shutil.rmtree(p,ignore_errors=True) for p in sys.argv[1:]]"

.PHONY: all no-omp debug clean distclean dirs
all: $(TARGET)

dirs:
	@$(MKDIRS) $(BUILDDIR) $(DATADIR)

$(TARGET): $(MAIN) $(HDR) | dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(SRCDIR) -o $@ $(MAIN)

no-omp: $(MAIN) $(HDR) | dirs
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $(TARGET) $(MAIN)

debug: $(MAIN) $(HDR) | dirs
	$(CXX) -O1 -g -std=c++17 -Wall -Wextra -fsanitize=address,undefined \
	  -I$(SRCDIR) -o $(DEBUGBIN) $(MAIN)

drift_probe: $(PROBESRC) $(SRCDIR)/model.hpp | dirs
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $(PROBE) $(PROBESRC)

clean:
	$(RMTREE) $(BUILDDIR)

distclean: clean
	$(RMTREE) $(DATADIR)
