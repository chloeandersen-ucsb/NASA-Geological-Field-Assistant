.PHONY: help setup run run-mock run-mock-ml run-conda clean check

# Project configuration
PYTHON := python3
PROJECT_ROOT := $(shell pwd)
LED_DISPLAY_DIR := $(PROJECT_ROOT)/led-display

# Detect platform
JETSON_DETECT := $(shell test -f /etc/nv_tegra_release && echo "jetson" || echo "other")
IS_JETSON := $(if $(filter jetson,$(JETSON_DETECT)),1,0)

# Conda configuration (update if needed)
CONDA_BASE := /Users/chloeandersen/miniconda3
CONDA_ENV := sage-ui
CONDA_SH := $(CONDA_BASE)/etc/profile.d/conda.sh

# Data directory configuration
ifeq ($(IS_JETSON),1)
	SAGE_STORE_DIR := /data/sage
else
	SAGE_STORE_DIR := $(LED_DISPLAY_DIR)/sage_data
endif

help:
	@echo "SAGE Project Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make setup       - Install dependencies on Jetson"
	@echo "  make run         - Start application in production mode"
	@echo "  make run-mock    - Start with mock services for testing"
	@echo "  make run-mock-ml - Start with mock ML, real voiceNotes"
	@echo "  make run-conda   - Start with conda environment (Mac)"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make check       - Verify all paths and dependencies"
	@echo ""
	@echo "Environment variables:"
	@echo "  SAGE_STORE_DIR           - Data storage directory"
	@echo "  SAGE_USE_MOCKS           - Set to 1 to use mock services (all)"
	@echo "  SAGE_USE_MOCK_ML         - Set to 1 to use mock ML only"
	@echo "  SAGE_ML_CLASSIFICATIONS_DIR - Override ML-classifications path"
	@echo "  SAGE_VOICE_TO_TEXT_DIR   - Override voiceNotes path"
	@echo "  SAGE_ROCKNET_WEIGHTS     - Override model weights path"
	@echo "  JETSON_PLATFORM          - Set to 1 to force Jetson mode"
	@echo ""
	@echo "Detected platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"

setup:
	@echo "Setting up SAGE project dependencies..."
	@echo "Platform: $(JETSON_DETECT)"
ifeq ($(IS_JETSON),1)
	@echo "Installing dependencies for Jetson..."
	@sudo apt-get update
	@sudo apt-get install -y python3-pip python3-dev
	@pip3 install PySide6
	@echo "Creating data directory: $(SAGE_STORE_DIR)"
	@sudo mkdir -p $(SAGE_STORE_DIR)
	@sudo chown $$USER:$$USER $(SAGE_STORE_DIR)
	@echo "Setup complete for Jetson"
else
	@echo "Installing dependencies for development platform..."
	@pip3 install PySide6
	@echo "Setup complete"
endif

run:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@cd $(LED_DISPLAY_DIR) && \
		export SAGE_STORE_DIR=$(SAGE_STORE_DIR) && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

run-mock:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Mock services enabled"
	@cd $(LED_DISPLAY_DIR) && \
		export SAGE_STORE_DIR=$(SAGE_STORE_DIR) && \
		export SAGE_USE_MOCKS=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py 2>&1

run-mock-ml:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Mock ML, Real voiceNotes"
	@cd $(LED_DISPLAY_DIR) && \
		export SAGE_STORE_DIR=$(SAGE_STORE_DIR) && \
		export SAGE_USE_MOCK_ML=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

run-conda:
	@if [ ! -f "$(CONDA_SH)" ]; then \
		echo "Error: Conda script not found at $(CONDA_SH)"; \
		echo "Please update CONDA_BASE in the Makefile to point to your conda installation."; \
		exit 1; \
	fi
	@echo "Starting SAGE application with conda environment..."
	@cd $(LED_DISPLAY_DIR) && \
		bash -c "source $(CONDA_SH) && conda activate $(CONDA_ENV) && \
		export SAGE_STORE_DIR=$(SAGE_STORE_DIR) && \
		export SAGE_USE_MOCKS=0 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py"

check:
	@echo "Checking SAGE project configuration..."
	@echo "Platform: $(JETSON_DETECT)"
	@echo ""
	@echo "Checking Python..."
	@$(PYTHON) --version || (echo "ERROR: Python3 not found" && exit 1)
	@echo "✓ Python found"
	@echo ""
	@echo "Checking project structure..."
	@test -d $(LED_DISPLAY_DIR) || (echo "ERROR: led-display directory not found" && exit 1)
	@test -d $(PROJECT_ROOT)/ML-classifications || (echo "WARNING: ML-classifications directory not found" && exit 1)
	@test -d $(PROJECT_ROOT)/voiceNotes || (echo "WARNING: voiceNotes directory not found" && exit 1)
	@test -f $(PROJECT_ROOT)/connector.py || (echo "ERROR: connector.py not found" && exit 1)
	@echo "✓ Project structure OK"
	@echo ""
	@echo "Checking Python dependencies..."
	@$(PYTHON) -c "import PySide6" 2>/dev/null || (echo "WARNING: PySide6 not installed (run 'make setup')" && exit 1)
	@echo "✓ PySide6 found"
	@echo ""
	@echo "Checking paths with connector..."
	@$(PYTHON) -c "import sys; sys.path.insert(0, '$(PROJECT_ROOT)'); import connector; \
		print('Project root:', connector.get_project_root()); \
		print('ML dir:', connector.get_ml_classifications_dir()); \
		print('voiceNotes dir:', connector.get_voice_to_text_dir()); \
		print('Data dir:', connector.get_data_store_dir()); \
		print('Is Jetson:', connector.is_jetson())" || (echo "ERROR: Path check failed" && exit 1)
	@echo "✓ Path resolution OK"
	@echo ""
	@echo "All checks passed!"

clean:
	@echo "Cleaning build artifacts..."
	@find $(PROJECT_ROOT) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find $(PROJECT_ROOT) -type f -name "*.pyc" -delete 2>/dev/null || true
	@find $(PROJECT_ROOT) -type f -name "*.pyo" -delete 2>/dev/null || true
	@find $(PROJECT_ROOT) -type f -name ".DS_Store" -delete 2>/dev/null || true
	@find $(PROJECT_ROOT) -type f -name "*.swp" -delete 2>/dev/null || true
	@find $(PROJECT_ROOT) -type f -name "*.swo" -delete 2>/dev/null || true
	@find $(PROJECT_ROOT) -type f -name "*~" -delete 2>/dev/null || true
	@find $(PROJECT_ROOT) -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find $(PROJECT_ROOT) -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find $(PROJECT_ROOT) -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"


#     @bash -c "source $(CONDA_SH) && conda activate $(CONDA_ENV) && export SAGE_USE_MOCKS=1 && python main.py &"