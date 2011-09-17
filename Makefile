egg: setup.py vivid/__init__.py
	python2 setup.py bdist_egg

vivid/__init__.py: python/vivid.py
	cp $< $@

