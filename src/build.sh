cd tensorrtx/retinaface
rm -rf build
mkdir build && cd build && cmake .. && make
cd ../../arcface
rm -rf build
mkdir build && cd build && cmake .. && make
cd ../../..
