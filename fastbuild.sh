g++ -O3 -fPIC -std=c++17 \
    -I"boost_install_1_85_0_3_13/include" -I"include/rdkit_2025_03_5" \
    -I"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
    -I"$(python3 -c 'import numpy; print(numpy.get_include())')" \
    -L"$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
    -L"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/rdkit/.dylibs" \
	-lboost_python313 \
	-lRDKitDataStructs.1 \
	-lpython3.13 \
    -shared smiles_fp.cpp -o _smiles_fp.so \
    #-Wl,-rpath,'$ORIGIN/../rdkit/.dylibs' \
    #-c smiles_fp.cpp -o _smiles_fp.o
install_name_tool -change /DLC/rdkit/.dylibs/libRDKitDataStructs.1.dylib @loader_path/../rdkit/.dylibs/libRDKitDataStructs.1.dylib _smiles_fp.so
install_name_tool -change /DLC/rdkit/.dylibs/libboost_python313.dylib @loader_path/../rdkit/.dylibs/libboost_python313.dylib _smiles_fp.so

g++ -O3 -fPIC -std=c++17 \
    -I"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
    -I"$(python3 -c 'import numpy; numpy.get_include()')" \
    -I"boost_install_1_81_0_3_10/include" \
    -L"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/rdkit.libs" \
    -shared smiles.cpp -o smiles.so \
    -Wl,-rpath,'$ORIGIN/../rdkit.libs'