# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /project/ev_sdk

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /project/ev_sdk/build

# Include any dependencies generated for this target.
include CMakeFiles/ji.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ji.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ji.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ji.dir/flags.make

CMakeFiles/ji.dir/src/ji.cpp.o: CMakeFiles/ji.dir/flags.make
CMakeFiles/ji.dir/src/ji.cpp.o: ../src/ji.cpp
CMakeFiles/ji.dir/src/ji.cpp.o: CMakeFiles/ji.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/ev_sdk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ji.dir/src/ji.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ji.dir/src/ji.cpp.o -MF CMakeFiles/ji.dir/src/ji.cpp.o.d -o CMakeFiles/ji.dir/src/ji.cpp.o -c /project/ev_sdk/src/ji.cpp

CMakeFiles/ji.dir/src/ji.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ji.dir/src/ji.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /project/ev_sdk/src/ji.cpp > CMakeFiles/ji.dir/src/ji.cpp.i

CMakeFiles/ji.dir/src/ji.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ji.dir/src/ji.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /project/ev_sdk/src/ji.cpp -o CMakeFiles/ji.dir/src/ji.cpp.s

CMakeFiles/ji.dir/src/sample_algorithm.cpp.o: CMakeFiles/ji.dir/flags.make
CMakeFiles/ji.dir/src/sample_algorithm.cpp.o: ../src/sample_algorithm.cpp
CMakeFiles/ji.dir/src/sample_algorithm.cpp.o: CMakeFiles/ji.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/ev_sdk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ji.dir/src/sample_algorithm.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ji.dir/src/sample_algorithm.cpp.o -MF CMakeFiles/ji.dir/src/sample_algorithm.cpp.o.d -o CMakeFiles/ji.dir/src/sample_algorithm.cpp.o -c /project/ev_sdk/src/sample_algorithm.cpp

CMakeFiles/ji.dir/src/sample_algorithm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ji.dir/src/sample_algorithm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /project/ev_sdk/src/sample_algorithm.cpp > CMakeFiles/ji.dir/src/sample_algorithm.cpp.i

CMakeFiles/ji.dir/src/sample_algorithm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ji.dir/src/sample_algorithm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /project/ev_sdk/src/sample_algorithm.cpp -o CMakeFiles/ji.dir/src/sample_algorithm.cpp.s

CMakeFiles/ji.dir/src/sample_detector.cpp.o: CMakeFiles/ji.dir/flags.make
CMakeFiles/ji.dir/src/sample_detector.cpp.o: ../src/sample_detector.cpp
CMakeFiles/ji.dir/src/sample_detector.cpp.o: CMakeFiles/ji.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/ev_sdk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ji.dir/src/sample_detector.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ji.dir/src/sample_detector.cpp.o -MF CMakeFiles/ji.dir/src/sample_detector.cpp.o.d -o CMakeFiles/ji.dir/src/sample_detector.cpp.o -c /project/ev_sdk/src/sample_detector.cpp

CMakeFiles/ji.dir/src/sample_detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ji.dir/src/sample_detector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /project/ev_sdk/src/sample_detector.cpp > CMakeFiles/ji.dir/src/sample_detector.cpp.i

CMakeFiles/ji.dir/src/sample_detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ji.dir/src/sample_detector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /project/ev_sdk/src/sample_detector.cpp -o CMakeFiles/ji.dir/src/sample_detector.cpp.s

# Object files for target ji
ji_OBJECTS = \
"CMakeFiles/ji.dir/src/ji.cpp.o" \
"CMakeFiles/ji.dir/src/sample_algorithm.cpp.o" \
"CMakeFiles/ji.dir/src/sample_detector.cpp.o"

# External object files for target ji
ji_EXTERNAL_OBJECTS =

libji.so: CMakeFiles/ji.dir/src/ji.cpp.o
libji.so: CMakeFiles/ji.dir/src/sample_algorithm.cpp.o
libji.so: CMakeFiles/ji.dir/src/sample_detector.cpp.o
libji.so: CMakeFiles/ji.dir/build.make
libji.so: 3rd/wkt_parser/libwktparser.so
libji.so: /usr/local/lib/libopencv_alphamat.so.4.5.1
libji.so: /usr/local/lib/libopencv_bioinspired.so.4.5.1
libji.so: /usr/local/lib/libopencv_freetype.so.4.5.1
libji.so: /usr/local/lib/libopencv_fuzzy.so.4.5.1
libji.so: /usr/local/lib/libopencv_hdf.so.4.5.1
libji.so: /usr/local/lib/libopencv_hfs.so.4.5.1
libji.so: /usr/local/lib/libopencv_img_hash.so.4.5.1
libji.so: /usr/local/lib/libopencv_intensity_transform.so.4.5.1
libji.so: /usr/local/lib/libopencv_line_descriptor.so.4.5.1
libji.so: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.1
libji.so: /usr/local/lib/libopencv_reg.so.4.5.1
libji.so: /usr/local/lib/libopencv_tracking.so.4.5.1
libji.so: /usr/local/lib/libopencv_highgui.so.4.5.1
libji.so: /usr/local/lib/libopencv_video.so.4.5.1
libji.so: /usr/local/lib/libopencv_videoio.so.4.5.1
libji.so: /usr/local/lib/libopencv_imgcodecs.so.4.5.1
libji.so: /usr/local/lib/libopencv_plot.so.4.5.1
libji.so: /usr/local/lib/libopencv_xphoto.so.4.5.1
libji.so: /usr/local/lib/libopencv_photo.so.4.5.1
libji.so: /usr/local/lib/libopencv_imgproc.so.4.5.1
libji.so: /usr/local/lib/libopencv_core.so.4.5.1
libji.so: CMakeFiles/ji.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/project/ev_sdk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libji.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ji.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ji.dir/build: libji.so
.PHONY : CMakeFiles/ji.dir/build

CMakeFiles/ji.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ji.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ji.dir/clean

CMakeFiles/ji.dir/depend:
	cd /project/ev_sdk/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /project/ev_sdk /project/ev_sdk /project/ev_sdk/build /project/ev_sdk/build /project/ev_sdk/build/CMakeFiles/ji.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ji.dir/depend

