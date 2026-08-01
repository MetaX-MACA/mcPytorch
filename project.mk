PROJECT_DEPS 	 	:=
PROJECT_NAME 	 	:= mcPytorch
PROJECT_MAKEFILE 	:= $(abspath $(lastword $(MAKEFILE_LIST)))
PROJECT_MK_PATH		:= $(patsubst %/, %, $(dir $(PROJECT_MAKEFILE)))

PROJECT_CONFIGURE	:= \
	echo "\n\t mcPytorch do not need common configure command! \n\t"

PROJECT_BUILD		:= \
	${MAKE} -f $(PROJECT_MK_PATH)/makefile.project.mk build_package

PROJECT_INSTALL		:= \
	echo "\n\t mcPytorch do not need common install command! \n\t"

PROJECT_CLEAN		:= \
	${MAKE} -f $(PROJECT_MK_PATH)/makefile.project.mk clean

include $(BUILD_COMMON)