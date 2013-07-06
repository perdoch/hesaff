:: How do I get pkg-config installed?
:: The difficulty in getting pkg-config installed is due its circular depency on glib. To install pkg-config, you should first install the GTK-runtime, the installer is found at http://sourceforge.net/project/showfiles.php?group_id=121075. The pkg-config binary can be obtained from http://www.gtk.org/download-windows.html. Place pkg-config.exe in your MinGW bin directory.
:: There are other pkg-config projects that don't have the circular dependency issue. They include:
:: pkgconf
:: pkg-config-lite

:: GTK Runtime
:: http://sourceforge.net/project/showfiles.php?group_id=121075

:: Pkg-config-lite 
:: wget http://sourceforge.net/projects/pkgconfiglite/files/
:: http://sourceforge.net/projects/pkgconfiglite/files/0.28-1/
:: set pkgconfiglite_location2=http://downloads.sourceforge.net/project/pkgconfiglite/0.28-1/pkg-config-lite-0.28-1_bin-win32.zip?r=&ts=1373150256&use_mirror=iweb

set pkgconfig_name=pkg-config-lite-0.28-1
set pkgconfig_zip=%pkgconfig_name%-win32.zip
set MINGW_BIN='C:\MinGW\bin'
set MINGW_SHARE='C:\MinGW\bin'
set pkg_config_dlsrc=http://downloads.sourceforge.net/project/pkgconfiglite/0.28-1/%pkgconfig_name%_bin-win32.zip

:: Download pkg-config-lite
wget %pkg_config_dlsrc%

:: Unzip and remove zipfile
unzip %pkgconfig_zip%
rm %pkgconfig_zip%

:: Install contents to MSYS
cp %pkgconfig_name%/bin/pkg-config.exe %MINGW_BIN%
cp -r %pkgconfig_name%/share/aclocal %MINGW_SHARE%
