
from pathlib import Path
import sys



base_path = Path(__file__).parent

file_path = (base_path / "../Flask").resolve()

sys.path.append(str(file_path))
# print(sys.path)

