.PHONY: setup run run-mock run-mock-ml run-mock-cam clean

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
	@pip3 install -r $(LED_DISPLAY_DIR)/requirements.txt
	@echo "Creating data directory: $(SAGE_STORE_DIR)"
	@sudo mkdir -p $(SAGE_STORE_DIR)
	@sudo chown $$USER:$$USER $(SAGE_STORE_DIR)
	@echo "Camera: nvgstcapture-1.0 is typically pre-installed with JetPack. Connect Arducam to CAM0 for capture."
	@echo "Setup complete for Jetson"
else
	@echo "Installing dependencies for development platform..."
	@pip3 install -r $(LED_DISPLAY_DIR)/requirements.txt
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
	@echo "Mode: Everything is mocked data"
	@cd $(LED_DISPLAY_DIR) && \
		export SAGE_STORE_DIR=$(SAGE_STORE_DIR) && \
		export SAGE_USE_MOCKS=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py 2>&1

run-mock-ml:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Mock ML, mock camera, real voiceNotes"
	@cd $(LED_DISPLAY_DIR) && \
		export SAGE_STORE_DIR=$(SAGE_STORE_DIR) && \
		export SAGE_USE_MOCK_ML=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

run-mock-cam:
	@echo "Platform: $(JETSON_DETECT)"
	@echo "Data directory: $(SAGE_STORE_DIR)"
	@echo "Mode: Real voice, real ML, sample image (no camera)"
	@cd $(LED_DISPLAY_DIR) && \
		export SAGE_STORE_DIR=$(SAGE_STORE_DIR) && \
		export SAGE_USE_SAMPLE_IMAGE=1 && \
		export JETSON_PLATFORM=$(IS_JETSON) && \
		$(PYTHON) main.py

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
	@rm -rf $(PROJECT_ROOT)/ML-classifications/camera-pipeline/captures 2>/dev/null || true
	@echo "Done!"

check:
	@echo "Checking SAGE project installation..."
	@echo "Platform: $(JETSON_DETECT)"
	@echo ""
	@ERRORS=0; \
	printf "Checking Python3... "; \
	if command -v $(PYTHON) >/dev/null 2>&1; then \
		PYTHON_VERSION=$$($(PYTHON) --version 2>&1); \
		echo "✓ $$PYTHON_VERSION"; \
	else \
		echo "✗ Python3 not found"; \
		ERRORS=$$((ERRORS + 1)); \
	fi; \
	printf "Checking pip3... "; \
	if command -v pip3 >/dev/null 2>&1; then \
		PIP_VERSION=$$(pip3 --version 2>&1); \
		echo "✓ $$PIP_VERSION"; \
	else \
		echo "✗ pip3 not found"; \
		ERRORS=$$((ERRORS + 1)); \
	fi; \
	if [ -f $(LED_DISPLAY_DIR)/requirements.txt ]; then \
		echo "Checking Python packages:"; \
		MISSING=0; \
		printf "  - PySide6... "; \
		if $(PYTHON) -c "import PySide6" >/dev/null 2>&1; then \
			echo "✓"; \
		else \
			echo "✗"; \
			MISSING=$$((MISSING + 1)); \
			ERRORS=$$((ERRORS + 1)); \
		fi; \
		printf "  - opencv-python-headless... "; \
		if $(PYTHON) -c "import cv2" >/dev/null 2>&1; then \
			echo "✓"; \
		else \
			echo "✗"; \
			MISSING=$$((MISSING + 1)); \
			ERRORS=$$((ERRORS + 1)); \
		fi; \
		printf "  - numpy... "; \
		if $(PYTHON) -c "import numpy" >/dev/null 2>&1; then \
			echo "✓"; \
		else \
			echo "✗"; \
			MISSING=$$((MISSING + 1)); \
			ERRORS=$$((ERRORS + 1)); \
		fi; \
		if [ $$MISSING -eq 0 ]; then \
			echo "  ✓ All required packages installed"; \
		fi; \
	else \
		echo "Checking Python packages... ✗ requirements.txt not found"; \
		ERRORS=$$((ERRORS + 1)); \
	fi; \
	printf "Checking data directory... "; \
	if [ -d "$(SAGE_STORE_DIR)" ]; then \
		echo "✓ $(SAGE_STORE_DIR) exists"; \
	else \
		echo "✗ $(SAGE_STORE_DIR) does not exist"; \
		echo "  Run 'make setup' to create it"; \
		ERRORS=$$((ERRORS + 1)); \
	fi; \
	if [ "$(IS_JETSON)" = "1" ]; then \
		printf "Checking nvgstcapture-1.0... "; \
		if command -v nvgstcapture-1.0 >/dev/null 2>&1; then \
			echo "✓ nvgstcapture-1.0 found"; \
		else \
			echo "⚠ nvgstcapture-1.0 not found (may need JetPack installation)"; \
		fi; \
	fi; \
	echo ""; \
	if [ $$ERRORS -eq 0 ]; then \
		echo "✓ All checks passed! Installation looks good."; \
		exit 0; \
	else \
		echo "✗ Found $$ERRORS issue(s). Run 'make setup' to install missing dependencies."; \
		exit 1; \
	fi
