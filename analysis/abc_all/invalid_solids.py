import json
import sys
from glob import glob
from pathlib import Path

if __name__ == '__main__':
    input_path = '/home/pete/brep_style/abc/mint-abc-uvnet-solid-status/derivative/*_solid_info.json'
    files = glob(input_path)

    for file in files:
        name = Path(file).stem[:-11]
        with open(file, 'rb') as f:
            solid_info = json.load(f)[0]
            if not solid_info['is_closed'] or solid_info['is_wire'] or solid_info['is_sheet']:
                print(name)
