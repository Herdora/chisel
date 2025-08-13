UNAME_S := $(shell uname -s)
TARGET_DIR := build
LIB_BASENAME := libgpucheck

.PHONY: all clean lib

all: $(TARGET_DIR) lib

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

ifeq ($(UNAME_S),Darwin)
LIB := $(TARGET_DIR)/$(LIB_BASENAME).dylib
lib: $(LIB)
$(LIB): gpu_check_mac.mm gpu_check.h | $(TARGET_DIR)
	clang++ -std=c++17 -fobjc-arc -dynamiclib -framework Metal -o $@ gpu_check_mac.mm
else
LIB := $(TARGET_DIR)/$(LIB_BASENAME).so
lib: $(LIB)
$(LIB): gpu_check_posix.cpp gpu_check.h | $(TARGET_DIR)
	$(CXX) -std=c++17 -fPIC -shared -ldl -O2 -o $@ gpu_check_posix.cpp
endif

clean:
	rm -rf $(TARGET_DIR)


