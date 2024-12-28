FEATER_DATA ?= $(shell realpath ./all_data)
# Update the ZENODO_ID to the latest version of the dataset
ZENODO_ID ?= 14235911

install: clean
		pip install build
		python -m build && pip install --force-reinstall -v dist/feater-0.0.1-py3-none-any.whl
		$(MAKE) clean

# micromamba install ambertools 

install_dependencies: 
	pip install git+https://github.com/miemiemmmm/SiESTA.git hilbertcurve matplotlib open3d 
	$(MAKE) dependency_training

# ViT/SwinTransformer and tensorboard while training
dependency_training: 
		pip install tensorboard transformers torch torchvision

clean: 
		rm -rf build dist feater.egg-info

einstall:
		pip install -e .

compile:
		cd src && make voxelize.so


download_baseline: 
		mkdir -p $(FEATER_DATA)
		cd $(FEATER_DATA) && wget https://zenodo.org/records/$(ZENODO_ID)/files/FEater_Baseline.tar.gz -O FEater_Baseline.tar.gz
		cd $(FEATER_DATA) && tar -xzvf FEater_Baseline.tar.gz

download_feater_single: 
		mkdir -p $(FEATER_DATA)
		cd $(FEATER_DATA) && wget https://zenodo.org/records/$(ZENODO_ID)/files/FEater_Single.tar.gz -O FEater_Single.tar.gz
		cd $(FEATER_DATA) && tar -xzvf FEater_Single.tar.gz 


download_feater_dual:
		mkdir -p $(FEATER_DATA)
		cd $(FEATER_DATA) && wget https://zenodo.org/records/$(ZENODO_ID)/files/FEater_Dual.tar.gz -O FEater_Dual.tar.gz
		cd $(FEATER_DATA) && tar -xzvf FEater_Dual.tar.gz 


download_minisets: 
		mkdir -p $(FEATER_DATA)
		cd $(FEATER_DATA) && wget https://zenodo.org/records/$(ZENODO_ID)/files/FEater_Mini200.tar.gz -O FEater_Mini200.tar.gz
		cd $(FEATER_DATA) && tar -xzvf FEater_Mini200.tar.gz
		cd $(FEATER_DATA) && wget https://zenodo.org/records/$(ZENODO_ID)/files/FEater_Mini400.tar.gz -O FEater_Mini400.tar.gz
		cd $(FEATER_DATA) && tar -xzvf FEater_Mini400.tar.gz
		cd $(FEATER_DATA) && wget https://zenodo.org/records/$(ZENODO_ID)/files/FEater_Mini800.tar.gz -O FEater_Mini800.tar.gz
		cd $(FEATER_DATA) && tar -xzvf FEater_Mini800.tar.gz
		

download_all:
		mkdir -p $(FEATER_DATA)
		$(MAKE) download_baseline
		$(MAKE) download_feater_single
		$(MAKE) download_feater_dual
		$(MAKE) download_minisets


# All source fragments
download_source: 
		mkdir -p $(FEATER_DATA)
		cd $(FEATER_DATA) && wget https://zenodo.org/records/12783988/files/FEater_Dual_PDBSRC.tar.gz -O FEater_Dual_PDBSRC.tar.gz
		cd $(FEATER_DATA) && wget https://zenodo.org/records/12783988/files/FEater_Single_PDBSRC.tar.gz -O FEater_Single_PDBSRC.tar.gz
