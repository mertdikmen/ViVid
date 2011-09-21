MODULES := __init__.py 
MODULES += _vivid.so

MODULES := $(addprefix vivid/, $(MODULES))

egg: setup.py $(MODULES) src
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

vivid/_vivid.so: src src/release/_vivid.so
	cp src/release/_vivid.so $@
