import sys
from pathlib import Path

# 레포 루트를 sys.path에 추가 → `import bench` 동작
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
