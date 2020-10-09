import os
import sys
from dataclasses import dataclass


@dataclass
class Project:
    windows_fmu_prefix = os.path.join("resources", "windows")
    linux_fmu_prefix = os.path.join("resources", "linux")

    def get_fmu_prefix(self):
        if "win" in sys.platform:
            return self.windows_fmu_prefix
        else:
            return self.linux_fmu_prefix
