.PHONY: launch seq neq clean pow2 map mod

# fwt.cu
launch: launchFWT.d
	dmd $< -of$@
	./$@

map: map.d
	dmd $< -of$@
	./$@

mod: mod.c
	gcc $< -o $@
	./$@

seq: sequency.d
	dmd $< -of$@
	./$@

neg: neg_pattern.c
	gcc $< -lm -o $@
	./$@

pow2: powersOf2.c
	gcc $< -o $@
	./$@

clean:
	rm -f *.o
	rm -f *.so
	rm -f launch neq seq pow2
