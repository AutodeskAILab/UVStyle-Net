import os
import sys

if __name__ == '__main__':
    abc_dir = '/home/pete/brep_style/abc/bin'
    invalid_solids = 'invalid_solids.txt'

    with open(invalid_solids, 'r') as file:
        for name in file.readlines():
            try:
                solid = f'{abc_dir}/{name[:-1]}.bin'
                print('removing:', solid)
                os.remove(solid)
            except Exception as e:
                print(e, file=sys.stderr)