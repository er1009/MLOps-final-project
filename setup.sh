pip install -r requirements.txt
git clone https://github.com/gretelai/gretel-synthetics.git
cd gretel-synthetics
pip uninstall tensorflow-estimator
pip install tensorflow-estimator==2.8
pip install gretel-synthetics
cd ..
git clone https://github.com/waico/tsad.git
git clone https://github.com/waico/SKAB.git
