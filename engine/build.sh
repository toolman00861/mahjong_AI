#!/bin/bash
set -ex
source /root/gpufree-data/mj/env.sh
rustc --version; cargo --version
cd $M/code/Mortal
git checkout -- libriichi
python $M/code/patch_libriichi.py $M/code/Mortal
cargo build -p libriichi --lib --release
cp target/release/libriichi.so $M/code/libriichi.so
cd $M/code && python -c "import libriichi; from libriichi.consts import obs_shape; print('libriichi ok', obs_shape(5))"
echo BUILD_DONE

