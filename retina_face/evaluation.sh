python test_widerface.py --trained_model .weights/mobilenet0.25_Final.pth --save_folder ./widerface_evaluate/mnet  --network mobile0.25


cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py --pred ./mnet