for i in {0..7};
do
    echo $(((10)**i));
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:. ./prog $((10**i)) 5 1
    echo ""
done

