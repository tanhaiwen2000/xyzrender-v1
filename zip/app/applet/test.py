import sys
try:
    import xyzrender.cli
    print(xyzrender.cli.__file__)
except Exception as e:
    print(e)
