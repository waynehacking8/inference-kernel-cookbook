NVCC      := /usr/local/cuda-12.8/bin/nvcc
CUDA_PATH := /usr/local/cuda-12.8

ARCH ?= sm_120

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) \
             -I$(CUDA_PATH)/include \
             -L$(CUDA_PATH)/lib64 \
             -lcublas

SRC_DIR := src
BUILD   := build

SOURCES := $(wildcard $(SRC_DIR)/*.cu)
TARGETS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD)/%,$(SOURCES))

.PHONY: all clean run

all: $(TARGETS)

$(BUILD)/%: $(SRC_DIR)/%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

run:
	@if [ -z "$(K)" ]; then \
		echo "Usage: make run K=01_flash_attention"; \
		echo "Available:"; \
		ls $(BUILD)/ 2>/dev/null || echo "  (none built — run 'make' first)"; \
	else \
		./$(BUILD)/$(K); \
	fi

run-all: all
	@for bin in $(sort $(TARGETS)); do \
		echo "\n========== $$(basename $$bin) =========="; \
		$$bin; \
	done
