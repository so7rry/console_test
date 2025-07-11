# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/ESP_IDF/5.3/v5.3/esp-idf/components/bootloader/subproject"
  "D:/esp-csi-master/esp-csi-master/examples/esp-radar/console_test/build/bootloader"
  "D:/esp-csi-master/esp-csi-master/examples/esp-radar/console_test/build/bootloader-prefix"
  "D:/esp-csi-master/esp-csi-master/examples/esp-radar/console_test/build/bootloader-prefix/tmp"
  "D:/esp-csi-master/esp-csi-master/examples/esp-radar/console_test/build/bootloader-prefix/src/bootloader-stamp"
  "D:/esp-csi-master/esp-csi-master/examples/esp-radar/console_test/build/bootloader-prefix/src"
  "D:/esp-csi-master/esp-csi-master/examples/esp-radar/console_test/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/esp-csi-master/esp-csi-master/examples/esp-radar/console_test/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/esp-csi-master/esp-csi-master/examples/esp-radar/console_test/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
