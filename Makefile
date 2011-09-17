egg: setup.py vivid vivid/__init__.py
	python2 setup.py bdist_egg

vivid:
	mkdir vivid

vivid/__init__.py: python/vivid.py
	cp $< $@

