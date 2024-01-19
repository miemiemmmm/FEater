install:
		python -m build && pip install --force-reinstall -v dist/feater-0.0.1-py3-none-any.whl

einstall:
		pip install --force-reinstall .

runpdb:
		pdbprocess -i /MieT5/BetaPose/data/complexes/complex_filelist.txt -d /media/yzhang/MieT72/Data/feater_test4/  \
		-wp 1 -wm 0 -ws 0 -wo 0 -prod 1 -r 0 -wn 40 -tn 1

compile:
		cd src && make voxelize.so





