.PHONY: help setup run run-raw run-base run-base-raw run-mock run-mock-ml clean check

# Project configuration
PYTHON := python3
PROJECT_ROOT := $(shell pwd)
LED_DISPLAY_DIR := $(PROJECT_ROOT)/led-display

# Detect platform
JETSON_DETECT := $(shell test -f /etc/nv_tegra_release && echo "jetson" || echo "other")
IS_JETSON := $(if $(filter jetson,$(JETSON_DETECT)),1,0)

# Data directory configuration
ifeq ($(IS_JETSON),1)
	SAGE_STORE_DIR := /data/sage
else
	SAGE_STORE_DIR := $(LED_DISPLAY_DIR)/sage_data
endif


setup:
	@echo "Setting up SAGE project dependencies..."
	@echo "Platform: $(JETSON_DETECT)"
ifeq ($(IS_JETSON),1)
	@echo "Installing dependencies for Jetson..."
	@sudo apt-get update
	@sudo apt-get install -y python3-pip python3-dev
	@pip3 install -r requirements.txt
	@echo "Downloading SAM model if needed"
	@cd $(PROJECT_ROOT) && $(PYTHON) rock-volume/download_sam.py
	@echo "Creating data directory: $(SAGE_STORE_DIR)"
	@sudo mkdir -p $(SAGE_STORE_DIR)
	@sudo chown $$USER:$$USER $(SAGE_STORE_DIR)
	@echo "Setup complete for Jetson"
else
	@echo "Installing dependencies for development platform..."
	@pip3 install -r requirements.txt
	@echo "Downloading SAM model if needed"
	@cd $(PROJECT_ROOT) && $(PYTHON) rock-volume/download_sam.py
	@echo "Setup complete"
endif

run:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@cd "$(LED_DISPLAY_DIR)" && \
		export SAGE_STORE_DIR="$(SAGE_STORE_DIR)" && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

run-raw:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Fine-tuned model (RAW ASR - No Post-Processing)"
	@cd "$(LED_DISPLAY_DIR)" && \
		export SAGE_STORE_DIR="$(SAGE_STORE_DIR)" && \
		export SAGE_RAW_ASR=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

run-base:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Base NeMo model"
	@cd "$(LED_DISPLAY_DIR)" && \
		export SAGE_STORE_DIR="$(SAGE_STORE_DIR)" && \
		export SAGE_USE_BASE_MODEL=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

run-base-raw:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Base NeMo model (RAW ASR - No Post-Processing)"
	@cd "$(LED_DISPLAY_DIR)" && \
		export SAGE_STORE_DIR="$(SAGE_STORE_DIR)" && \
		export SAGE_USE_BASE_MODEL=1 && \
		export SAGE_RAW_ASR=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

run-mock:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Mock services enabled"
	@cd "$(LED_DISPLAY_DIR)" && \
		export SAGE_STORE_DIR="$(SAGE_STORE_DIR)" && \
		export SAGE_USE_MOCKS=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py 2>&1

run-mock-ml:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Mock ML, Real voiceNotes"
	@cd "$(LED_DISPLAY_DIR)" && \
		export SAGE_STORE_DIR="$(SAGE_STORE_DIR)" && \
		export SAGE_USE_MOCK_ML=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

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