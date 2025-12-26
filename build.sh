# Clone, build, and install boost 1.81.0
cpu_cores() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sysctl -n hw.logicalcpu
    else
        nproc
    fi
}

rdkit="$1"

py=$(python3 --version | awk '{print $2}')
py_major=$(echo $py | cut -d. -f1)
py_minor=$(echo $py | cut -d. -f2)
py_underscore="${py_major}_${py_minor}"

mkdir -p build
cd build

if [ ! -d "rdkit_${rdkit}" ]; then
    git clone https://github.com/rdkit/rdkit --config "advice.detachedHead=false" --branch $rdkit --depth 1 rdkit_${rdkit}
    echo "RDKit cloned!"
else
    echo "RDKit already cloned!"
fi

boost=$(grep 'RDK_BOOST_VERSION' "rdkit_${rdkit}/CMakeLists.txt" | head -n 1 | awk -F'"' '{print $2}')
boost=$(python3 -c "
from packaging.version import parse
if parse('${py}') < parse('3.12'):
    print('${boost}')
else:
    print(max(parse('${boost}'), parse('1.85.0')))")

echo "Python: ${py} RDKit: ${rdkit} Boost: ${boost}"
b_major=$(echo $boost | cut -d. -f1)
b_minor=$(echo $boost | cut -d. -f2)
b_patch=$(echo $boost | cut -d. -f3)
boost_underscore="${b_major}_${b_minor}_${b_patch}"


if [ ! -d "boost_${boost_underscore}" ]; then
        if [ ! -f "boost_${boost_underscore}.tar.gz" ]; then
            echo "Downloading Boost..."
            curl "https://archives.boost.io/release/${boost}/source/boost_${boost_underscore}.tar.gz" --output "boost_${boost_underscore}.tar.gz"
            echo "Boost downloaded!"
        else
            echo "Boost already downloaded!"
        fi
        echo "Extracting Boost..."
        tar -xvf "boost_${boost_underscore}.tar.gz"
        echo "Boost extracted!"
    else
        echo "Boost already extracted!"
    fi

boost_install="boost_install_${boost_underscore}_${py_underscore}"

if [ ! -d "${boost_install}/include" ]; then  
    cd "boost_${boost_underscore}"
    echo "Bootstrapping boost..."
    rm -rf project-config.jam b2 bin.v2
    ./bootstrap.sh  --prefix="../${boost_install}" \
                    --with-python=$(which python) \
                    --with-libraries=python,system,serialization,iostreams,program_options

    echo "Boost bootstrapped!"

    echo "Building and installing boost..."
    ./b2 install    -j$(cpu_cores) \
                    include=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
                    variant=release

    echo "Boost installed!"
    cd .. # base dir
else
    echo "Boost already installed!"
fi

if [ -d "${boost_install}/include/boost" ]; then
    mkdir -p ../include
    rsync -a "${boost_install}/include/boost" ../include/boost
fi

# Build RDKit
rdkit_install="rdkit_install_${rdkit}_${boost_underscore}_${py_underscore}"

# never do this - we only need code
if [ ! -d "${rdkit_install}/include" ]; then
    echo "Configuring RDKit..."
    cd "rdkit_${rdkit}"
    rm -rf build
    mkdir build
    cd build
    
    libpython=$(python3 -c "import sysconfig, pathlib; p=sysconfig.get_config_var('LIBDIR'); v=sysconfig.get_config_var('LDVERSION'); print(pathlib.Path(p)/f'libpython{v}.dylib')")
    if [[ "$OSTYPE" == "darwin"* ]]; then
        linkerflags="${libpython}"
    fi

    cmake ..    -DRDK_INSTALL_INTREE=OFF \
                -DRDK_INSTALL_STATIC_LIBS=OFF \
                -DCMAKE_INSTALL_PREFIX="../../${rdkit_install}" \
                -DBoost_ROOT="$PWD/../../${boost_install}" \
                -DCMAKE_SHARED_LINKER_FLAGS="${linkerflags}"
                # -DPython3_EXECUTABLE=$(which python) \
                # -DPython3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
                # -DPython3_LIBRARY="${libpython}" \
                
    echo "RDKit configured!"

    echo "Installing RDKit..."
    make install -j$(cpu_cores)
    echo "RDKit installed!"
    cd .. # rdkit
    # python -m Scripts.gen_rdkit_stubs ../rdkit_install_${rdkit}_${boost_underscore}_${py_underscore}
    cd .. # base dir
else
    echo "RDKit already installed!"
fi

if [ -d "${rdkit_install}/include/rdkit" ]; then
    mkdir -p ../include
    rsync -a "${rdkit_install}/include/rdkit" ../include
fi
