egg: setup.py vivid vivid/__init__.py src
	python2 setup.py bdist_egg

.PHONY: src
src:
	$(MAKE) -C src

vivid:
	mkdir vivid

clean:
	$(RM) src/release/*.so
	$(RM) src/release/*.o

vivid/__init__.py: python/vivid.py
	cp $< $@
