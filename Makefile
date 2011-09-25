MODULES := __init__.py 
MODULES += vivid_kmeans.py
MODULES += flexible_filter.py
MODULES += cv_conversions.py
MODULES += _vivid.so

MODULES := $(addprefix vivid/, $(MODULES))

egg: $(MODULES) src setup.py
	python2 setup.py bdist_egg

vivid:
	mkdir vivid

.PHONY: src
src:
	$(MAKE) -C src

.PHONY: test
test:
	$(MAKE) -C src test

vivid/__init__.py: python/vivid.py
	cp $< $@

vivid/%.py: python/%.py
	cp $< $@

vivid/_vivid.so: src src/release/_vivid.so
	cp src/release/_vivid.so $@

clean:
	$(RM) src/release/*.so
	$(RM) src/release/*.o
	$(RM) src/release/*.co
