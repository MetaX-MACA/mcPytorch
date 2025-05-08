PYTHON_VERSION ?= 3.8.16
PYTORCH_MACA_COMPILER_PATH := $(if $(MACA_CLANG_PATH),$(MACA_CLANG_PATH),"")
PROJECT_BUILDDIR 	:= $(BUILDDIR)/framework
PROJECT_MAKEFILE 	:= $(abspath $(lastword $(MAKEFILE_LIST)))
PROJECT_MK_PATH		:= $(patsubst %/, %, $(dir $(PROJECT_MAKEFILE)))
PROJECT_EXTERND_OPS 	:=

build_package:
	mkdir -p $(DEFAULT_INSTALL_DIR)/wheel/;				\
	bash $(PROJECT_MK_PATH)/maca_tools/build_and_run.sh        	\
		--maca_path $(DEFAULT_INSTALL_DIR)                      \
		--maca_compiler_path $(PYTORCH_MACA_COMPILER_PATH)      \
		--conda_env_dst_python_version $(PYTHON_VERSION)        \
		--py_setup_cmd bdist_wheel                              \
		--remove_cache                                          \
		--clean_conda_env_dst                                   \
		--build_type "$(BUILD_TYPE)"                            \
		--maca_version "$(MACA_VERSION)"                        \
		--max_jobs "$(NUM_JOB)"                                 \
		--dst_wheel_dir_path $(DEFAULT_INSTALL_DIR)/wheel/      \
		--verbose; \

	mkdir -p $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/; \
	cp $(PROJECT_MK_PATH)/build/bin/* $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/; \
	cp -rf $(PROJECT_MK_PATH)/maca_tests $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/; \
	cp -rf $(PROJECT_MK_PATH)/maca_samples $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/; \
	cp -rf $(PROJECT_MK_PATH)/test $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/; \
	cp -rf $(PROJECT_MK_PATH)/tools $(DEFAULT_INSTALL_DIR)/test/acl/bin/pytorch/

clean:
	bash $(PROJECT_MK_PATH)/maca_tools/build_and_run.sh        	\
		--maca_path $(DEFAULT_INSTALL_DIR)                      \
		--remove_cache                                          \
		--skip_build                                            \
		--verbose