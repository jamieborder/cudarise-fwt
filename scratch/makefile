.PHONY: launch seq neq clean

# fwt.cu
launch: launchFWT.d
	dmd $< -of$@
	./$@

seq: sequency.d
	dmd $< -of$@
	./$@

neg: neg_pattern.c
	gcc $< -lm -o $@
	./$@

clean:
	rm -f *.o neg
