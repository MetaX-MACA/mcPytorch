#PROJECT_DEPS := mcDnn mcBlas
PROJECT_DEPS :=

mcPytorch:PRIVATE_BUILD_DIR :=$(TARGET_BUILD_DIR)/framework/mcPytorch
mcPytorch:PRIVATE_MK_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))
mcPytorch:PRIVATE_MK_PATH := $(patsubst %/, %, $(dir $(PRIVATE_MK_FILE)))
mcPytorch:PRIVATE_PYTORCH_MACA_COMPILER_PATH := $(if $(MACA_CLANG_PATH),$(MACA_CLANG_PATH),"")
mcPytorch:PYTHON_VERSION ?= 3.8.16
mcPytorch: $(PROJECT_DEPS)
	echo "DEFAULT_INSTALL_DIR: "$(DEFAULT_INSTALL_DIR)
	echo "TARGET_BUILD_DIR: "$(TARGET_BUILD_DIR)
	echo "PRIVATE_BUILD_DIR: "$(PRIVATE_BUILD_DIR)
	echo "PRIVATE_MK_PATH: "$(PRIVATE_MK_PATH)
	echo "MACA_CLANG_PATH: "$(MACA_CLANG_PATH)
	echo "BUILD_TYPE: "$(BUILD_TYPE)
	echo "PRIVATE_PYTORCH_MACA_COMPILER_PATH: "$(PRIVATE_PYTORCH_MACA_COMPILER_PATH)
	echo "MACA_VERSION: "$(MACA_VERSION)
	echo "NUM_JOB: "$(NUM_JOB)
	echo "PYTHON_VERSION: "$(PYTHON_VERSION)

	mkdir -p $(DEFAULT_INSTALL_DIR)/wheel/

	bash $(PRIVATE_MK_PATH)/maca_tools/build_and_run.sh        \
		--maca_path $(DEFAULT_INSTALL_DIR)                               \
		--maca_compiler_path $(PRIVATE_PYTORCH_MACA_COMPILER_PATH)       \
		--conda_env_dst_python_version $(PYTHON_VERSION)                 \
		--py_setup_cmd bdist_wheel                                       \
		--remove_cache                                                   \
		--clean_conda_env_dst                                            \
		--build_type "$(BUILD_TYPE)"                                     \
		--maca_version "$(MACA_VERSION)"                                 \
		--max_jobs "$(NUM_JOB)"                                         \
		--dst_wheel_dir_path $(DEFAULT_INSTALL_DIR)/wheel/               \
		--verbose

	mkdir -p $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/
	cp $(PRIVATE_MK_PATH)/build/bin/* $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/
	cp -rf $(PRIVATE_MK_PATH)/maca_tests $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/
	cp -rf $(PRIVATE_MK_PATH)/maca_samples $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/
	cp -rf $(PRIVATE_MK_PATH)/test $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/
	cp -rf $(PRIVATE_MK_PATH)/tools $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/


mcPytorch-clean:PRIVATE_BUILD_DIR :=$(TARGET_BUILD_DIR)/framework/mcPytorch
mcPytorch-clean:PRIVATE_MK_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))
mcPytorch-clean:PRIVATE_MK_PATH := $(patsubst %/, %, $(dir $(PRIVATE_MK_FILE)))
mcPytorch-clean:
	bash $(PRIVATE_MK_PATH)/maca_tools/build_and_run.sh        \
		--maca_path $(DEFAULT_INSTALL_DIR)                               \
		--remove_cache                                                   \
		--skip_build                                                     \
		--verbose
