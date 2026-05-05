NVCC     = nvcc
ARCH     = sm_86       # RTX 3050 (Ampere). Change to sm_89 for RTX 4060.
FLAGS    = -O3 -std=c++17 -arch=$(ARCH)
SRC      = src
BUILD    = build

.PHONY: all benchmark nn_naive nn_optimized hello clean

all: benchmark nn_naive nn_optimized

$(BUILD):
	mkdir -p $(BUILD)

benchmark: $(BUILD)
	$(NVCC) $(FLAGS) $(SRC)/benchmark.cu -o $(BUILD)/benchmark

nn_naive: $(BUILD)
	$(NVCC) $(FLAGS) $(SRC)/nn_naive.cu -o $(BUILD)/nn_naive

nn_optimized: $(BUILD)
	$(NVCC) $(FLAGS) $(SRC)/nn_optimized.cu -o $(BUILD)/nn_optimized

hello: $(BUILD)
	$(NVCC) $(FLAGS) $(SRC)/hello.cu -o $(BUILD)/hello

clean:
	rm -rf $(BUILD)
