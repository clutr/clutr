import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)
print(os.pardir)
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)